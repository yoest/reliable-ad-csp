import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Type
from .image_enc import Encoder, Decoder
from .time_enc import SinusoidalEncoding

from .model_wrapper import ConditionalGlowWrapper


class ForecastModel(nn.Module):

    def __init__(self, model_args: dict):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.ts_input_dim = model_args['ts_input_dim']
        self.ts_hidden_dim = model_args['ts_hidden_dim']
        self.ts_proj_dim = model_args['ts_proj_dim']
        self.ts_num_layers = model_args['ts_num_layers']
        self.seq_len = model_args['seq_len']
        self.indim = model_args['indim']
        self.cdim = model_args['cdim']
        self.zdim = model_args['zdim']
        self.time_emb_in = model_args['time_emb_in']
        self.time_emb_out = model_args['time_emb_out']
        self.difffeature = model_args['difffeature']
        self.n_flows = model_args['n_flows']
        self.n_blocks = model_args['n_blocks']

        if model_args['time_series_arch'] == 'lstm':
            self.time_series_arch = nn.LSTM(input_size=self.ts_input_dim,
                                hidden_size=self.ts_hidden_dim,
                                proj_size=self.ts_proj_dim,
                                num_layers=self.ts_num_layers,
                                batch_first=True)
        else:
            raise ValueError(f"Unknown time_series_arch: {model_args['time_series_arch']}")

        self.image_encoder = Encoder(input_size=self.indim, rep_dim=self.zdim, channels=1)
        if model_args.get('with_reconstruction', False):
            self.image_decoder = Decoder(input_size=self.indim, rep_dim=self.zdim, channels=1)

        self.wrapper = ConditionalGlowWrapper(
            num_channels=1,
            num_flows=self.n_flows,
            n_blocks=self.n_blocks,
            img_shape=(1, self.indim, self.indim),
            device=model_args['device']
        )

        self.time_encoder_in = SinusoidalEncoding(emb_dim = self.time_emb_in)
        self.time_encoder_out = SinusoidalEncoding(emb_dim = self.time_emb_out)

    def square_of_dim(self, dim):
        assert np.sqrt(dim) % 1 < 1e-5, "Dim must be a square number"
        return int(np.sqrt(dim))

    def compute_context(self, feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature):
        indiff, outdiff = difffeature.split("_")

        # Image encoding
        batch_size, seq_length, c, h, w = feature_data.shape
        feature_data = feature_data.view(batch_size * seq_length, c, h, w)
        feature_data = self.image_encoder(feature_data)
        feature_data = feature_data.view(batch_size, seq_length, -1)

        # Time encoding: Feature
        if indiff in ['tau', 'all']:
            batch_size, seq_length, d = feature_diff.shape
            feature_diff = feature_diff.view(batch_size * seq_length, d)
            feature_diff = self.time_encoder_in(feature_diff)
            feature_diff = feature_diff.view(batch_size, seq_length, -1)

        if indiff in ['delta', 'all']:
            batch_size, seq_length, d = feature_startdiff.shape
            feature_startdiff = feature_startdiff.view(batch_size * seq_length, d)
            feature_startdiff = self.time_encoder_in(feature_startdiff)
            feature_startdiff = feature_startdiff.view(batch_size, seq_length, -1)

        diff = None
        if indiff == 'all':
            diff = torch.cat([feature_diff, feature_startdiff], dim =-1)
        elif indiff == 'tau':
            diff = feature_diff
        elif indiff == 'delta':
            diff = feature_startdiff

        if diff is not None:
            history, _ = self.time_series_arch(torch.cat([feature_data, diff], dim=-1))
        else:
            history, _ = self.time_series_arch(feature_data)

        batch_size, seq_length, d = history.shape
        history = history[:,-1,:].contiguous()

        # Time encoding: Target
        if outdiff in ['tau', 'all']:
            target_diff = self.time_encoder_out(target_diff)

        if outdiff in ['delta', 'all']:
            target_startdiff = self.time_encoder_out(target_startdiff)

        diff2 = None
        if outdiff == 'all':
            diff2 = torch.cat([target_diff, target_startdiff], dim =-1)
        elif outdiff == 'tau':
            diff2 = target_diff
        elif outdiff == 'delta':
            diff2 = target_startdiff

        if diff2 is not None:
            out = torch.cat([history, diff2], dim=-1)
        else:
            out = history

        return out

    def get_condition(self, feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature):
        out = self.compute_context(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)
        out = self.redimension_context(out)
        return out

    def forward(self,
                feature_data: Type[torch.tensor],
                feature_diff: Type[torch.tensor],
                feature_startdiff: Type[torch.tensor],
                target_data: Type[torch.tensor],
                target_diff: Type[torch.tensor],
                target_startdiff: Type[torch.tensor],
                difffeature: str = 'diff'
                ):
        out = self.get_condition(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)
        return self.wrapper(x=target_data, c=out)

    def redimension_context(self, context):
        assert np.sqrt(context.shape[1]) % 1 < 1e-5, "Context size must be a square number"
        context_size = int(np.sqrt(context.shape[1]))

        return context.view(-1, 1, context_size, context_size)

    def sample(self, feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature, n_samples: int = 1):
        out = self.get_condition(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)
        
        samples, nlls = [], []
        batch_size = target_diff.shape[0]
        for i in range(batch_size):
            out_batch = out[i].unsqueeze(0) if out is not None else None
            out_batch = out_batch.repeat(n_samples, 1, 1, 1)

            y_sample, nll = self.wrapper.sample(c=out_batch, n_samples=n_samples)
            samples.append(y_sample)
            nlls.append(nll)

        samples = torch.stack(samples)
        nlls = torch.stack(nlls)

        # Add the channel dimension
        samples = samples.unsqueeze(2)

        return samples, nlls

    def reconstruct(self,
                feature_data: Type[torch.tensor],
                feature_diff: Type[torch.tensor],
                feature_startdiff: Type[torch.tensor],
                target_data: Type[torch.tensor],
                target_diff: Type[torch.tensor],
                target_startdiff: Type[torch.tensor],
                difffeature: str = 'diff'
                ):
        out = self.get_condition(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)

        # Reconstruct the input
        return self.wrapper.reconstruct(x=target_data, c=out)
    
    def get_negative_log_likelihood(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature):
        out = self.get_condition(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)
        _, nll = self.wrapper(x=target_data, c=out) 
        return nll
    
    def get_density(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature):
        nll = self.get_negative_log_likelihood(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature)
        return torch.exp(-nll)
    
    def get_latent(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature):
        out = self.get_condition(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature)            
        return self.wrapper.get_latent(x=target_data, c=out)