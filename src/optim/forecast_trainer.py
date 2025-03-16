import logging
import torch
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from pathlib import Path
from utils.config import Config
from typing import Type
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from sklearn.manifold import TSNE

from dataset.csp_dataset import CSPDataset
from model import ForecastModel, Encoder, Autoencoder

from threshold.risk_function import FPRFunction, FNRFunction, F1Function, WeightedFPRFNRFunction
from threshold.ltt import LTT
from threshold.thresholding import F1ScoreThresholding, GMeanThresholding, ZScoreThresholding
from threshold.score_cache import ScoreCache



class ForecastTrainer():

    def __init__(self, config: Type[Config]):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.optimizer_name = config['optimizer_name']
        self.early_stopping = config['early_stopping']
        self.lr = config['learning_rate']
        self.n_epochs = config['epochs']
        self.patience = config['patience']
        self.lr_milestones = config['learning_rate_milestones']
        self.batch_size = config['batch_size']
        self.weight_decay = config['weight_decay']
        self.n_jobs_dataloader = config['num_workers']
        self.difffeature = config.settings['difffeature']
        self.device = config['device']

        self.model_name = 'densityad_' + config['csp'] + '_' + config['experiment']

        self.model_path = config['model_path']
        self.seq_len = config['seq_len']
        self.output_path = config['output_path']

        self.init_model()

    def init_model(self):
        indim = self.config['indim']
        cdim = self.config['cdim']
        zdim = self.config['zdim']
        timedim = self.config['timedim']

        assert np.sqrt(zdim) % 1 < 1e-5, 'zdim must be a perfect square'

        indiff, outdiff = self.difffeature.split("_")
        assert indiff in ['no', 'tau', 'delta', 'all']

        if indiff == 'no':
            ts_input_dim = zdim
        elif indiff == 'tau' or indiff == 'delta':
            ts_input_dim = zdim + timedim
        else:
            ts_input_dim = zdim + timedim * 2

        assert outdiff in ['no', 'tau', 'delta', 'all']
        if outdiff == 'no':
            assert np.sqrt(cdim) % 1 < 1e-5, 'cdim must be a perfect square'
            assert np.sqrt(cdim) >= np.sqrt(zdim) // 2, 'square of cdim must be at least half of square of zdim'
        elif outdiff == 'tau' or outdiff == 'delta':
            assert np.sqrt(cdim + timedim) % 1 < 1e-5, 'cdim must be a perfect square'
            assert np.sqrt(cdim + timedim) >= np.sqrt(zdim) // 2, 'square of cdim and timedim must be at least half of square of zdim'
        else:
            assert np.sqrt(cdim + timedim * 2) % 1 < 1e-5, 'cdim must be a perfect square'
            assert np.sqrt(cdim + timedim * 2) >= np.sqrt(zdim) // 2, 'square of cdim and timedim must be at least half of square of zdim'

        ts_proj_dim = cdim
        ts_hidden_dim = cdim * 2

        model_args = {
            'ts_input_dim': ts_input_dim,
            'ts_hidden_dim': ts_hidden_dim,
            'ts_proj_dim': ts_proj_dim,
            'ts_num_layers': self.config.get('llayers', 40),
            'seq_len': self.seq_len,
            'indim': indim,
            'cdim': cdim,
            'zdim': zdim,
            'time_emb_in': timedim,
            'time_emb_out': timedim,
            'difffeature': self.difffeature,
            'time_series_arch': self.config.get('time_series_arch', 'lstm'),
            'device': self.device,
            'n_flows': self.config['n_flows'],
            'n_blocks': self.config['n_blocks'],
        }

        self.logger.info(model_args)
        self.model = ForecastModel(model_args).to(self.device)

        # If running for testing always load the trained model
        if not self.config['train_forecast']:
            self.logger.info('Loading forecast model')
            self.load_model(self.model_name)

        if self.config['pretrained_ae']:
            self.logger.info('Loading pretrained AE for forecast model')
            self.model.image_encoder = self.init_network_weights_from_pretraining(self.model.image_encoder, self.config['pretrained_ae'], input_size=indim, rep_dim=zdim)

    def train(self, dataset: CSPDataset):
        self.logger.info('Starting training...')
        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        self.model.to(device=self.device)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        train_loss, val_loss = [], []
        stop_nan_loss = False

        for epoch in range(self.n_epochs):
            if epoch in self.lr_milestones:
                self.logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            train_loss_batch = []
            val_loss_batch = []

            # --- Training ---
            for input in train_loader:
                optimizer.zero_grad()

                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, _ = input
                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                            feature_diff.to(self.device),\
                                                                                                            feature_startdiff.to(self.device),\
                                                                                                            target_data.to(self.device),\
                                                                                                            target_diff.to(self.device),\
                                                                                                            target_startdiff.to(self.device)

                z, nll = self.model(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature=self.difffeature)
                if torch.isnan(nll).any():
                    print("NaN detected in training loss")
                    stop_nan_loss = True
                    break

                loss = torch.mean(nll)
                loss.backward()

                # Limit the magnitude of gradients to prevent exploding updates
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                optimizer.step()

                train_loss_batch.append(loss.item())
            avg_train_loss = sum(train_loss_batch) / len(train_loss_batch) if len(train_loss_batch) > 0 else 0
            train_loss.append(avg_train_loss)

            scheduler.step()

            if stop_nan_loss:
                self.logger.info('Stopping training due to NaN loss')
                self.model = copy.deepcopy(best_model)
                break

            # --- Validation ---
            val_scores = []
            val_labels = []

            with torch.no_grad():
                for input in val_loader:
                    feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, _, labels, _ = input
                    feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                                feature_diff.to(self.device),\
                                                                                                                feature_startdiff.to(self.device),\
                                                                                                                target_data.to(self.device),\
                                                                                                                target_diff.to(self.device),\
                                                                                                                target_startdiff.to(self.device)

                    if feature_data.shape[0] == 0:
                        continue

                    z, nll = self.model(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature=self.difffeature)
                    if torch.isnan(nll).any():
                        raise ValueError("NaN detected in validation loss")

                    val_scores.extend(nll.cpu().data.numpy())
                    val_labels.extend(labels.cpu().data.numpy())

                    loss = torch.mean(nll)

                    val_loss_batch.append(loss.item())
            avg_val_loss = sum(val_loss_batch) / len(val_loss_batch) if len(val_loss_batch) > 0 else 0
            
            val_auc = roc_auc_score(np.array(val_labels), np.array(val_scores))
            val_aupr = average_precision_score(np.array(val_labels), np.array(val_scores))

            val_loss.append(avg_val_loss)

            self.logger.info("Epoch %d: train loss %.4f, val loss %.4f, val AUC %.4f, val AUPR %.4f" % (epoch+1, avg_train_loss, avg_val_loss, val_auc, val_aupr))

            # Implement early stopping
            start_epoch = 10 if not self.config['debug'] else 0

            if self.early_stopping and epoch > start_epoch:  # Start checking after 20 epochs
                if epoch == start_epoch + 1:
                    # min_loss = avg_val_loss
                    max_aupr = val_aupr
                    best_model = copy.deepcopy(self.model)
                    patience_cnt = 0
                    continue

                # if avg_val_loss < min_loss:
                if val_aupr > max_aupr:
                    # min_loss = avg_val_loss
                    max_aupr = val_aupr
                    patience_cnt = 0

                    best_model = copy.deepcopy(self.model)
                else:
                    patience_cnt +=1
                    if patience_cnt == self.patience:
                        self.logger.info('Training stops at {} epoch'.format(epoch+1))
                        break

        self.logger.info('Finished training.')
        if self.early_stopping:
            self.model = copy.deepcopy(best_model)

        self.plot_loss(train_loss, val_loss)
        self.save_model(self.model_name)

    def plot_loss(self, train_loss: list, val_loss: list, skip_first: bool = True):
        if skip_first:
            train_loss = train_loss[1:]
            val_loss = val_loss[1:]

        plt.figure(figsize=(10, 5))
        plt.title(self.config['experiment'])
        plt.plot(train_loss, label='Train')
        plt.plot(val_loss, label='Validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('NLL')
        plt.tight_layout()

        fig_path = Path(self.output_path)
        if not Path.exists(fig_path):
            fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path.joinpath('losses.pdf'), bbox_inches='tight')

    def save_model(self, name: Type[str]):
        """ Method to save the trained models
        Args:
            name (str): name of the saved model file
        """

        if not Path.exists(Path(self.model_path)):
            Path.mkdir(Path(self.model_path), parents=True, exist_ok=True)

        model_dict = self.model.state_dict()

        export_path = Path(self.model_path).joinpath(name + '.tar')

        torch.save(model_dict, export_path)
        self.logger.info("Saving model to file")

    def load_model(self, name: Type[str]):
        """ Method to load saved model
        Args:
            name (str): name of the saved model file
        """
        self.logger.info("Loading model from file")
        import_path = Path(self.config['model_path'])

        if not Path.exists(import_path):
            print(import_path)
            raise ValueError('Model checkpoint path is invalid')

        # self.model.load_state_dict(torch.load(import_path, map_location=torch.device('cpu')))

        load_dict = torch.load(import_path, map_location=torch.device('cpu'))

        forcast_dict = self.model.state_dict()
        updated_dict = {k: v for k, v in load_dict.items() if k in forcast_dict}

        self.model.load_state_dict(updated_dict)

    def init_network_weights_from_pretraining(self, image_encoder: Type[Encoder], pretrain_path: Type[str], input_size: int, rep_dim: int):
        self.logger.info("Initialising Image encoder from pretrained AE")

        pretrained_model = Autoencoder(input_size=input_size, rep_dim=rep_dim, channels=1)
        if not Path.exists(Path(pretrain_path)):
            raise ValueError('Model checkpoint path is invalid')

        pretrained_model.load_state_dict(torch.load(Path(pretrain_path)))
        pretrained_model_dict = pretrained_model.state_dict()

        image_encoder_dict = image_encoder.state_dict()
        updated_dict = {k: v for k, v in pretrained_model_dict.items() if k in image_encoder_dict}
        image_encoder_dict.update(updated_dict)
        image_encoder.load_state_dict(image_encoder_dict)

        return image_encoder
    
    def test(self, dataset):
        self.logger.info('Start testing.')
        self.model.eval()

        if not Path.exists(Path(self.config['output_path'])):
            Path.mkdir(Path(self.config['output_path']), parents=True)

        _, val_loader, test_loader = dataset.loaders(batch_size = self.config['batch_size'], num_workers = self.config['num_workers'])

        # == Sanity check that the model is working ==
        contexts, contexts_nll = [], []
        normal_max_nll, abnormal_min_nll = -np.inf, np.inf
        normal_max_nll_img, abnormal_min_nll_img = None, None
        already_sampled = False

        with torch.no_grad():
            for input in tqdm(test_loader, desc="Sanity check"):
                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, _, labels, _ = input
                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                            feature_diff.to(self.device),\
                                                                                                            feature_startdiff.to(self.device),\
                                                                                                            target_data.to(self.device),\
                                                                                                            target_diff.to(self.device),\
                                                                                                            target_startdiff.to(self.device)

                z, nll = self.model(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature=self.difffeature)

                context = self.model.compute_context(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature=self.difffeature)
                contexts.append(context)
                contexts_nll.append(nll)

                # Find the normal and abnormal samples with the highest and lowest NLL
                for i in range(len(nll)):
                    if labels[i] == 0:
                        if nll[i] > normal_max_nll:
                            normal_max_nll = nll[i].item()
                            normal_max_nll_img = target_data[i]
                    else:
                        if nll[i] < abnormal_min_nll:
                            abnormal_min_nll = nll[i].item()
                            abnormal_min_nll_img = target_data[i]

                # Test the sampling
                if not already_sampled:
                    samples, _ = self.model.sample(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature=self.difffeature, n_samples=6)

                    # Plot samples
                    plot_samples = samples[0, :6, 0, :, :]
                    self.plot_samples(plot_samples, target_data[0, :, :])
                    already_sampled = True

                    # Plot reconstruction sanity check
                    recon_images = self.model.reconstruct(feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, difffeature=self.difffeature)
                    base_images, recon_images = target_data[:6], recon_images[:6]
                    self.plot_reconstructed_images(base_images, recon_images)

        contexts = torch.cat(contexts, dim=0)
        contexts_nll = torch.cat(contexts_nll, dim=0)
        self.plot_context(contexts, contexts_nll)
        if abnormal_min_nll_img is not None and normal_max_nll_img is not None:
            self.plot_normal_abnormal_misclassified(normal_max_nll, normal_max_nll_img, abnormal_min_nll, abnormal_min_nll_img)

        # == Scores ==
        calib_scores = ScoreCache(self.config, self.model, self.difffeature)
        calib_scores.compute_scores(val_loader, Path(self.config['output_path']).joinpath('calib_scores.json'))

        test_scores = ScoreCache(self.config, self.model, self.difffeature)
        test_scores.compute_scores(test_loader, Path(self.config['output_path']).joinpath('test_scores.json'))

        # == AUROC and AUPR ==
        test_labels, test_cur_scores = test_scores.labels, test_scores.log_dens_scores
        test_labels, test_cur_scores = np.array(test_labels), np.array(test_cur_scores)

        results = {
            'auroc': roc_auc_score(test_labels, test_cur_scores),
            'aupr': average_precision_score(test_labels, test_cur_scores)
        }
        with open(Path(self.config['output_path']).joinpath('test_metrics_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # == Thresholding ==
        alpha, delta = self.config['alpha'], self.config['delta']
        risk_fct = None
        if self.config['risk_fct'] == 'fpr':
            risk_fct = FPRFunction()
        elif self.config['risk_fct'] == 'fnr': 
            risk_fct = FNRFunction()
        elif self.config['risk_fct'] == 'w_fnr_fpr':
            risk_fct = WeightedFPRFNRFunction()
        elif self.config['risk_fct'] == 'f1':
            risk_fct = F1Function()
        else:
            raise ValueError('Risk function not implemented')
        
        # Load thresholds from given files
        thresholds = None
        if self.config.get('thresholds_path', None):
            with open(Path(self.config['thresholds_path']), 'r') as f:
                thresholds = json.load(f)
            self.logger.info('Thresholds loaded from file')
        
        f1_thresh = F1ScoreThresholding(thresholds=(thresholds['f1']['lam_l'], thresholds['f1']['lam_h']) if thresholds else None)
        if not thresholds:
            f1_thresh.fit(calib_scores.log_dens_scores, calib_scores.labels)

        gmean_thresh = GMeanThresholding(thresholds=(thresholds['gmean']['lam_l'], thresholds['gmean']['lam_h']) if thresholds else None)
        if not thresholds:
            gmean_thresh.fit(calib_scores.log_dens_scores, calib_scores.labels)

        zscore_thresh = ZScoreThresholding(thresholds=(thresholds['zscore']['lam_l'], thresholds['zscore']['lam_h']) if thresholds else None)
        if not thresholds:
            zscore_thresh.fit(calib_scores.log_dens_scores, calib_scores.labels)

        n_space_lambdas = 400
        dr_ltt = LTT(risk_fct, alpha, delta, n_space_lambdas=n_space_lambdas, verbose=True, thresholds=(thresholds['dr_ltt']['lam_l'], thresholds['dr_ltt']['lam_h']) if thresholds else None)
        if not thresholds:
            dr_ltt.fit(calib_scores.log_dens_scores, calib_scores.labels)

        l_ltt = LTT(risk_fct, alpha, delta, n_space_lambdas=n_space_lambdas, verbose=True, thresholds=(thresholds['l_ltt']['lam_l'], thresholds['l_ltt']['lam_h']) if thresholds else None)
        if not thresholds:
            l_ltt.fit(calib_scores.latent_norm_scores, calib_scores.labels)

        ltt_hdr = LTT(risk_fct, alpha, delta, n_space_lambdas=n_space_lambdas, verbose=True, thresholds=(thresholds['ltt_hdr']['lam_l'], thresholds['ltt_hdr']['lam_h']) if thresholds else None)
        if not thresholds:
            ltt_hdr.fit(calib_scores.hdr_scores, calib_scores.labels)

        # Add all the results
        with open(Path(self.config['output_path']).joinpath(f'{self.config["risk_fct"]}_thresholds_results.json'), 'w') as f:
            json.dump({
                'f1': {
                    'lam_l': f1_thresh.lam_l.item(), 'lam_h': f1_thresh.lam_h.item(),
                    'results': f1_thresh.test(test_scores.log_dens_scores, test_scores.labels)
                },
                'gmean': {
                    'lam_l': gmean_thresh.lam_l.item(), 'lam_h': gmean_thresh.lam_h.item(),
                    'results': gmean_thresh.test(test_scores.log_dens_scores, test_scores.labels)
                },
                'zscore': {
                    'lam_l': zscore_thresh.lam_l, 'lam_h': zscore_thresh.lam_h,
                    'results': zscore_thresh.test(test_scores.log_dens_scores, test_scores.labels)
                },
                'dr_ltt': {
                    'lam_l': dr_ltt.lam_l.item(), 'lam_h': dr_ltt.lam_h.item(),
                    'results': dr_ltt.test(test_scores.log_dens_scores, test_scores.labels)
                },
                'l_ltt': {
                    'lam_l': l_ltt.lam_l.item(), 'lam_h': l_ltt.lam_h.item(),
                    'results': l_ltt.test(test_scores.latent_norm_scores, test_scores.labels)
                },
                'ltt_hdr': {
                    'lam_l': ltt_hdr.lam_l.item(), 'lam_h': ltt_hdr.lam_h.item(),
                    'results': ltt_hdr.test(test_scores.hdr_scores, test_scores.labels)
                }
            }, f, indent=4)

        self.logger.info('Finished testing.')
        self.logger.info(f'If you want more details about the results, run the thresholding and related notebooks.')
        self.logger.info('Done.')

    def plot_context(self, contexts: torch.Tensor, contexts_nll: torch.Tensor):
        # Use T-SNE to reduce the dimensionality of the context
        context_tsne = TSNE(n_components=2).fit_transform(contexts.cpu().data.numpy())

        fig, ax = plt.subplots(7, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [8, 1, 1, 1, 1, 1, 1]})
        fig.suptitle(self.config['difffeature'] + f' (Seed {self.config["seed"]})')

        # Plotting the context reduced to 2D
        im = ax[0].scatter(context_tsne[:, 0], context_tsne[:, 1], c=contexts_nll.cpu().data.numpy(), cmap='RdYlGn_r')
        cbar = fig.colorbar(im, ax=ax[0])
        cbar.set_label('NLL')

        # Plot some high and low NLL samples
        n = 3
        high_nll_idx = torch.argsort(contexts_nll, descending=True)[:n]

        for i in range(n):
            reshaped_context = contexts[high_nll_idx[i]].view(1, -1)
            im = ax[i+1].imshow(reshaped_context.cpu().data.numpy(), cmap='cividis')
            ax[i+1].axis('off')
        ax[1].set_title('High NLL')

        low_nll_idx = torch.argsort(contexts_nll, descending=False)[:n]
        for i in range(n):
            reshaped_context = contexts[low_nll_idx[i]].view(1, -1)
            im = ax[i+4].imshow(reshaped_context.cpu().data.numpy(), cmap='cividis')
            ax[i+4].axis('off')
        ax[4].set_title('Low NLL')

        plt.tight_layout(h_pad=0)

        fig_path = Path(self.config['output_path'])
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path.joinpath('context.pdf'), bbox_inches='tight')

    def plot_samples(self, samples: torch.Tensor, target_latent_data: torch.Tensor):
        fig, ax = plt.subplots(1, 1 + len(samples), figsize=(4 * len(samples), 4))
        fig.suptitle(self.config['difffeature'] + f' (Seed {self.config["seed"]})')

        # Plot the target latent data
        im = ax[0].imshow(target_latent_data.squeeze().cpu().data.numpy(), cmap='RdYlGn_r')
        fig.colorbar(im, ax=ax[0])
        ax[0].axis('off')
        ax[0].set_title('Target Latent')

        for i in range(1, len(samples) + 1):
            im = ax[i].imshow(samples[i - 1].cpu().numpy(), cmap='RdYlGn_r')
            fig.colorbar(im, ax=ax[i])
            ax[i].axis('off')
            ax[i].set_title(f'Sample {i}')

        plt.tight_layout()
        
        fig_path = Path(self.config['output_path'])
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path.joinpath('samples.pdf'), bbox_inches='tight')

    def plot_reconstructed_images(self, base_images, reconstructed_images):
        assert len(base_images) == len(reconstructed_images), 'Number of base images and reconstructed images must be the same'

        fig, ax = plt.subplots(3, len(base_images), figsize=(4 * len(base_images), 8))

        for i in range(len(base_images)):
            im = ax[0, i].imshow(base_images[i].squeeze().cpu().data.numpy(), cmap='RdYlGn_r')
            ax[0, i].axis('off')
            ax[0, i].set_title(f'Base {i}')
            fig.colorbar(im, ax=ax[0, i])

            im = ax[1, i].imshow(reconstructed_images[i].squeeze().cpu().data.numpy(), cmap='RdYlGn_r')
            ax[1, i].axis('off')
            ax[1, i].set_title(f'Reconstructed {i}')
            fig.colorbar(im, ax=ax[1, i])

            # Plot diff
            diff = base_images[i] - reconstructed_images[i]
            im = ax[2, i].imshow(diff.squeeze().cpu().data.numpy(), cmap='cividis')
            ax[2, i].axis('off')
            ax[2, i].set_title(f'Diff {i}')
            fig.colorbar(im, ax=ax[2, i])

        plt.tight_layout()
        
        fig_path = Path(self.config['output_path'])
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path.joinpath('reconstructed_images.pdf'), bbox_inches='tight')

    def plot_normal_abnormal_misclassified(self, normal_max_nll, normal_max_nll_img, abnormal_min_nll, abnormal_min_nll_img):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(self.config['difffeature'] + f' (Seed {self.config["seed"]})')

        # Plot the normal max nll image
        im = ax[0].imshow(normal_max_nll_img.squeeze().cpu().data.numpy(), cmap='RdYlGn_r')
        fig.colorbar(im, ax=ax[0])
        ax[0].axis('off')
        ax[0].set_title(f'Normal with Max NLL: {round(normal_max_nll, 4)}')

        # Plot the abnormal min nll image
        im = ax[1].imshow(abnormal_min_nll_img.squeeze().cpu().data.numpy(), cmap='RdYlGn_r')
        fig.colorbar(im, ax=ax[1])
        ax[1].axis('off')
        ax[1].set_title(f'Abnormal with Min NLL: {round(abnormal_min_nll, 4)}')

        plt.tight_layout()
        
        fig_path = Path(self.config['output_path'])
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path.joinpath('normal_abnormal_misclassified.pdf'), bbox_inches='tight')