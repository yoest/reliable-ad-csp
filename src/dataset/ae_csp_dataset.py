import torch
import numpy as np
import logging
import json
import pickle
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader
from utils.config import Config

DISABLE_TQDM = False
RECTIFYSEQ = True
EPS = 1e-5


class AECSPDataset():

    def __init__(self, config: Config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_set = AECSPSet(config = self.config, transform=transform)
        self.val_set = AECSPSet(config = self.config, transform=transform, mode = 'val')
        self.test_set = AECSPSet(config = self.config, transform=transform, mode = 'test')

        self.logger.info(f"Train samples: {len(self.train_set)} Val samples: {len(self.val_set)} Test samples: {len(self.test_set)}")
        self.logger.info('Dataset Configured')

    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0):
        """Initialise data loaders

        Args:
            batch_size (int): batch size
            shuffle (bool, optional): shuffle the data samples. Defaults to True.
            num_workers (int, optional): number of concurrent workers. Defaults to 0.

        Returns:
            train_dataLoader (DataLoader): train data loader
            val_dataLoader (DataLoader): validation data loader
            test_dataLoader (DataLoader): test data loader
        """
        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_dataLoader, val_dataLoader, test_dataLoader


class AECSPSet(AECSPDataset):

    def __init__(self, config: Config, transform: transforms = None, mode: str = 'train'):
        self.config = config
        self.indim = self.config['indim']
        self.root = self.config['data_path']
        self.device = self.config['device']
        self.transform = transform
        self.mode = mode
        self.datamodelist = self.config['datamodelist']
        self.seq_len = self.config['seq_len']

        self.datapath = Path(self.root)
        self.data_seq, self.label_seq = self.__make_dataset__()

    def __make_dataset__(self):
        # load from JSON file
        count_path = Path(self.config["data_path"]).parent
        with open(count_path.joinpath(f'{self.config["csp"]}.json'), 'r') as fp:
            cur_labels = json.load(fp)

        n_normal = len(cur_labels['normal'])
        n_anomalous = len(cur_labels['anomalous'])
        train_val_split = self.config['train_val_test_split']

        train_n_normal = int(n_normal * train_val_split[0])
        train_n_anomalous = 0

        val_n_normal = int(n_normal * train_val_split[1])
        val_n_anomalous = n_anomalous // 2

        test_n_normal = n_normal - train_n_normal - val_n_normal
        test_n_anomalous = n_anomalous - train_n_anomalous - val_n_anomalous

        assert train_n_normal > 0 and val_n_normal > 0 and test_n_normal > 0, f"Invalid split: {train_n_normal} {val_n_normal} {test_n_normal}"
        assert val_n_anomalous > 0 and test_n_anomalous > 0, f"Invalid split: {val_n_anomalous} {test_n_anomalous}"

        if self.mode == 'train':
            samples = cur_labels['normal'][:train_n_normal].copy()
            samples.extend(cur_labels['anomalous'][:train_n_anomalous].copy())
            labels = None
        elif self.mode == 'val':
            samples = cur_labels['normal'][train_n_normal:train_n_normal + val_n_normal].copy()
            samples.extend(cur_labels['anomalous'][train_n_anomalous:train_n_anomalous + val_n_anomalous].copy())
            labels = [0]*val_n_normal
            labels.extend([1]*val_n_anomalous)
        elif self.mode == 'test':
            samples = cur_labels['normal'][train_n_normal + val_n_normal:].copy() 
            samples.extend(cur_labels['anomalous'][train_n_anomalous + val_n_anomalous:].copy())
            labels = [0]*test_n_normal
            labels.extend([1]*test_n_anomalous)

        # Reduce the number of samples for development mode
        if self.config['debug']:
            samples = samples[:100]
            if labels is not None:
                labels = labels[:100]

        return samples, labels

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        Tensor = torch.LongTensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.datapath.joinpath(str(self.data_seq[idx]) + '.pkl')

        with open(file, 'rb') as fp:
            image = pickle.load(fp)['image']
        image = image.astype(np.float32)
        
        channels = 1
        image = np.repeat(image[:, :, np.newaxis], channels, axis=2)

        if self.transform:
            image = self.transform(image)

        if self.mode == "train" or self.mode == "val":
            assert 0 <= image.min() and image.max() <= 1, f"Problem with normalization. Image min: {image.min()} max: {image.max()}"

        if self.mode == "train":
            return image, Tensor([int(self.data_seq[idx])])
        else:
            return image, Tensor([int(self.data_seq[idx])]), Tensor([self.label_seq[idx]])