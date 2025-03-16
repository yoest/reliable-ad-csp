from utils.config import Config
from model.forecast_model import ForecastModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import json
import os


class ScoreCache:

    def __init__(self, cfg: Config, model: ForecastModel, difffeature: str):
        self.cfg = cfg
        self.model = model
        self.difffeature = difffeature
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = cfg.get('verbose', True)

        self.log_dens_scores = []
        self.latent_norm_scores = []
        self.hdr_scores = []
        self.labels = []
        self.timestamps = []

    def compute_scores(self, loader: DataLoader, output_path: str):
        if os.path.exists(output_path):
            print(f"Loading scores from {output_path}")
            with open(output_path, 'r') as f:
                scores = json.load(f)
                self.log_dens_scores = torch.tensor(scores['log_dens_scores'])
                self.latent_norm_scores = torch.tensor(scores['latent_norm_scores'])
                self.hdr_scores = torch.tensor(scores['hdr_scores'])
                self.labels = torch.tensor(scores['labels'])
            return

        for full_inputs in tqdm(loader, disable=not self.verbose):
            feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, cur_timestamps, cur_labels, _ = full_inputs
            feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                        feature_diff.to(self.device),\
                                                                                                        feature_startdiff.to(self.device),\
                                                                                                        target_data.to(self.device),\
                                                                                                        target_diff.to(self.device),\
                                                                                                        target_startdiff.to(self.device)

            log_dens_score = self._get_log_density_score(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff)
            latent_norm_score = self._get_latent_norm_score(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff)
            hdr_score = self._get_hdr_score(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff)

            self.log_dens_scores.extend(log_dens_score.detach().cpu().numpy().tolist())
            self.latent_norm_scores.extend(latent_norm_score.detach().cpu().numpy().tolist())
            self.hdr_scores.extend(hdr_score.detach().cpu().numpy().tolist())
            self.labels.extend(cur_labels.detach().cpu().numpy().flatten().tolist())
            self.timestamps.extend(cur_timestamps.detach().cpu().numpy().flatten().tolist())

        # Save the scores
        with open(output_path, 'w') as f:
            json.dump({
                'log_dens_scores': self.log_dens_scores,
                'latent_norm_scores': self.latent_norm_scores,
                'hdr_scores': self.hdr_scores,
                'labels': self.labels,
                'timestamps': self.timestamps
            }, f, indent=4)

        self.log_dens_scores = torch.tensor(self.log_dens_scores)
        self.latent_norm_scores = torch.tensor(self.latent_norm_scores)
        self.hdr_scores = torch.tensor(self.hdr_scores)
        self.labels = torch.tensor(self.labels)
    
    def _get_log_density_score(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff):
        return self.model.get_negative_log_likelihood(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, self.difffeature)
    
    def _get_latent_norm_score(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff):
        latent = self.model.get_latent(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, self.difffeature)
        latent = latent.detach().cpu()
        return torch.norm(latent, dim=1)

    def _get_hdr_score(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff):
        return self._hpd(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff)
    
    def _hpd(self, feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff):
        """ This code has been taken from https://github.com/Vekteur/multi-output-conformal-regression and 
            is licensed under the MIT License. 
        """
        densities = []
        for _ in range(3):
            _, nlls = self.model.sample(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, self.difffeature, n_samples=40)
            nlls = nlls.detach().cpu()
            cur_densities = torch.exp(-nlls)
            densities.append(cur_densities)
        densities = torch.cat(densities, dim=1)
        density_x = self.model.get_density(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, self.difffeature)
        density_x = density_x.detach().cpu()

        cdf = self._fast_empirical_cdf(densities, density_x)
        hpd = 1 - cdf
        return hpd
    
    def _fast_empirical_cdf(self, a, b):
        """ This code has been taken from https://github.com/Vekteur/multi-output-conformal-regression and
            is licensed under the MIT License.
        """
        assert a.dim() == 2 and b.dim() >= 1 and a.shape[0] == b.shape[0]
        b_shape = b.shape
        if b.dim() == 1:
            # The naive implementation is faster for this case.
            cdf = (a <= b[:, None]).float().mean(dim=-1)
        else:
            a = torch.sort(a, dim=1)[0]
            # These operations are needed because the first N - 1 dimensions of a and b
            # have to be the same.
            view = (a.shape[0],) + (1,) * len(b.shape[1:-1]) + (a.shape[1],)
            repeat = (1,) + b.shape[1:-1] + (1,)
            a = a.view(*view).repeat(*repeat)
            cdf = torch.searchsorted(a, b.contiguous(), side='right') / a.shape[-1]
        
        # We move n back to the last dimension.
        assert cdf.shape == b_shape
        return cdf