import logging
import time
import torch
import copy

from model.image_enc import Autoencoder
from utils.config import Config
from pathlib import Path
from typing import Type


class AETrainer():

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
        self.device = config['device']

        self.model_name = 'pretrainedae_' + config['csp'] + '_seed' + str(config['seed']) + '_indim' + str(config['indim']) + '_zdim' + str(config['zdim'])
        
        self.model_path = config['model_path']
        self.seq_len = config['seq_len']
        self.output_path = config['output_path']

        self.init_model()

    def init_model(self):
        indim = self.config['indim']
        zdim = self.config['zdim']

        self.model = Autoencoder(input_size=indim, rep_dim=zdim, channels=1).to(self.device)

        # If running for testing always load the trained model
        if not self.config['train_forecast']:
            self.logger.info('Loading forecast model')
            self.load_model(self.model_name)

    def train(self, dataset):
        self.logger.info('Starting training...')
        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        self.model.to(device=self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        start_time = time.time()
        for epoch in range(self.n_epochs):
            train_loss_batch = []
            val_loss_batch = []

            for input in train_loader:
                optimizer.zero_grad()

                feature_data, _ = input
                feature_data = feature_data.to(self.device)

                pred_data = self.model(feature_data)

                scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))

                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                train_loss_batch.append(loss.item())

            avg_train_loss = sum(train_loss_batch) / len(train_loss_batch)

            with torch.no_grad():
                for input in val_loader:

                    feature_data, _, _ = input
                    feature_data = feature_data.to(self.device)

                    pred_data = self.model(feature_data)

                    scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))
                    loss = torch.mean(scores)

                    val_loss_batch.append(loss.item())

            avg_val_loss = sum(val_loss_batch) / len(val_loss_batch)

            self.logger.info("Epoch %d: train MSE %.4f, val MSE %.4f" % (epoch, avg_train_loss, avg_val_loss))

            # Implement early stopping
            if self.early_stopping:
                if epoch == 0:
                    min_loss = avg_val_loss
                    best_model = copy.deepcopy(self.model)
                    patience_cnt = 0
                    continue

                if avg_val_loss < min_loss:
                    min_loss = avg_val_loss
                    patience_cnt = 0

                    best_model = copy.deepcopy(self.model)
                else:
                    patience_cnt +=1
                    if patience_cnt == self.patience:
                        self.logger.info('Training stops at {} epoch'.format(epoch+1))
                        break

        train_time = time.time() - start_time

        self.logger.info('Finished training. Total time: %.3f' % train_time)

        if self.early_stopping:
            self.model = copy.deepcopy(best_model)

        model_path = self.save_model(self.model_name)

        return model_path

    def test(self, dataset):
        self.logger.info('Starting testing...')
        _, _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        self.model.to(device=self.device)
        self.model.eval()

        test_mse = []
        with torch.no_grad():
            for input in test_loader:

                feature_data, _, _ = input
                feature_data = feature_data.to(self.device)

                pred_data = self.model(feature_data)

                scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))
                loss = torch.mean(scores)

                test_mse.append(loss.item())

        avg_test_mse = sum(test_mse) / len(test_mse)
        self.logger.info("Test MSE %.4f" % avg_test_mse)
        self.logger.info('Finished testing.')

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
        self.logger.info(f"Saving model to file {export_path.as_posix()}")

        return export_path

    def load_model(self, name: Type[str]):
        """ Method to load saved model
        Args:
            name (str): name of the saved model file
        """
        self.logger.info("Loading model from file")
        import_path = Path(self.model_path).joinpath(name + '.tar')

        if not Path.exists(import_path):
            raise ValueError('Model checkpoint path is invalid')

        load_dict = torch.load(import_path, map_location=torch.device('cpu'))

        forcast_dict = self.model.state_dict()
        updated_dict = {k: v for k, v in load_dict.items() if k in forcast_dict}

        self.model.load_state_dict(updated_dict)