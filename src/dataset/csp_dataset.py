import torch
import numpy as np
import logging
import json
import pickle
import datetime

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from utils.config import Config

DISABLE_TQDM = False
RECTIFYSEQ = True
EPS = 1e-5


class CSPDataset:

    def __init__(self, config: Config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.train_set = CSPSet(config = self.config, logger=self.logger, mode = 'train')
        self.val_set = CSPSet(config = self.config, logger=self.logger, mode = 'val')
        self.test_set = CSPSet(config = self.config, logger=self.logger, mode = 'test')

        self.logger.info(f"Train samples: {len(self.train_set)} Val samples: {len(self.val_set)} Test samples: {len(self.test_set)}")
        self.logger.info('Dataset Configured')

    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0):
        """ Initialise data loaders

        Args:
            batch_size (int): batch size
            shuffle (bool, optional): shuffle the data samples. Defaults to True.
            num_workers (int, optional): number of concurrent workers. Defaults to 0.

        Returns:
            train_dataLoader (DataLoader): train data loader
            val_dataLoader (DataLoader): validation data loader
            test_dataLoader (DataLoader): test data loader
        """
        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # No need to shuffle test data for a fair evaluation between models
        return train_dataLoader, val_dataLoader, test_dataLoader


class CSPSet(CSPDataset):

    def __init__(self, config: Config, logger: logging.Logger, mode: str = 'train'):
        self.config = config
        self.logger = logger

        self.root = self.config['data_path']
        self.datapath = Path(self.root) 

        self.device = self.config['device']
        self.seq_len = self.config['seq_len']
        self.indim = self.config['indim']
        self.mode = mode

        self.data_seq, self.diff_seq, self.startdiff_seq, self.label_seq = self.__make_dataset__()

    def __make_dataset__(self):
        cache_path = Path(self.config['cache_path'])
        cache_path.mkdir(parents=True, exist_ok=True)

        if not self.config['load_cache']:
            self.logger.info(f'Generating cache for mode {self.mode}...')
            
            file_list = {
                'image_path': [],
                'timestamps': [],
                'mode': [],
                'days': []
            }

            p = self.datapath.glob('*.*')
            imgpaths = [x for x in p if x.is_file()]

            file_list['image_path'].extend(imgpaths)
            file_list['timestamps'].extend([int(x.stem) for x in imgpaths])
            file_list['days'].extend([datetime.date.fromtimestamp(int(x.stem)/1000) for x in imgpaths])

            # Sort samples based on timestamp
            timestamps = []
            imgpaths = []
            days = []

            for i,x,j in tqdm(sorted(zip(file_list['timestamps'], file_list['days'], file_list['image_path'])), disable=DISABLE_TQDM, desc='[CACHE] Sorting'):
                timestamps.append(i)
                imgpaths.append(j)
                days.append(x)

            file_list = {
                'image_path': imgpaths,
                'timestamps': timestamps,
                'days': np.array(days)
            }

            # Free Space
            timestamps = []
            imgpaths = []
            days = []

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
            self.logger.info(f"Train normal: {train_n_normal} Train anomalous: {train_n_anomalous} Val normal: {val_n_normal} Val anomalous: {val_n_anomalous} Test normal: {test_n_normal} Test anomalous: {test_n_anomalous}")

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

            # initialize arrays
            data_seq, diff_seq, startdiff_seq, label_seq  = [], [], [], []

            # loop over samples
            for sample_idx, sample in enumerate(tqdm(samples, disable=DISABLE_TQDM, desc='[CACHE] Processing')):
                # get the index for the current sample
                sample_ind = np.where(np.array(file_list['timestamps']) == int(sample))[0][0]

                if sample_ind < self.seq_len + 1:
                    continue

                # get seq_len + 2 samples where 1 (previous) + seq_len (context) + 1 (target)
                currtimeseq = file_list['timestamps'][sample_ind - (self.seq_len + 1) : sample_ind + 1]
                currtimediff, currentstartdiff = [], []

                # get the date of the current sample
                seq_date = datetime.date.fromtimestamp(currtimeseq[-1]/1000)
                seq_datetime = datetime.datetime.combine(seq_date, datetime.time(8, 0, 0))

                # compute the interarrival times
                VALIDSEQ = True
                validfrom = 0
                INCORRPREV = False

                for idx, ts in enumerate(currtimeseq):
                    if idx == 0:
                        if (datetime.date.fromtimestamp(ts/1000) == seq_date) and (datetime.datetime.fromtimestamp(ts/1000) >= seq_datetime):
                            prevtimestamp = ts
                            continue
                        else:
                            INCORRPREV = True
                            continue
                    if (datetime.date.fromtimestamp(ts/1000) == seq_date) and (datetime.datetime.fromtimestamp(ts/1000) >= seq_datetime):
                        if INCORRPREV:
                            currtimediff.append(EPS)
                            INCORRPREV = False
                            validfrom = idx - 1
                        else:
                            currtimediff.append((datetime.datetime.fromtimestamp(ts/1000) - datetime.datetime.fromtimestamp(prevtimestamp/1000)).total_seconds())
                    else:
                        currtimediff.append(EPS)
                        INCORRPREV = True
                        VALIDSEQ = False

                    prevtimestamp = ts

                # List of file paths
                currdateseq = file_list['image_path'][sample_ind - self.seq_len : sample_ind + 1]

                # Either rectify by repeating or discard invalid sequences
                if not VALIDSEQ:
                    if RECTIFYSEQ:
                        for idx in range(len(currdateseq)):
                            if idx < validfrom:
                                currdateseq[idx] = currdateseq[validfrom]
                                currtimeseq[idx + 1] = currtimeseq[validfrom + 1]
                    else:
                        continue

                data_seq.append(currdateseq)
                diff_seq.append(currtimediff)

                # Get interval between the current timestamp and the beginning of the day
                starttime_idx = np.where(file_list['days'] == seq_date)[0][0]
                starttime = datetime.datetime.fromtimestamp(file_list['timestamps'][starttime_idx]/1000)

                while starttime < seq_datetime and datetime.date.fromtimestamp(file_list['timestamps'][starttime_idx]/1000) == seq_date:
                    starttime_idx += 1
                    starttime = datetime.datetime.fromtimestamp(file_list['timestamps'][starttime_idx]/1000)

                # print(f'Start of {seq_date} is at {datetime.date.fromtimestamp(file_list["timestamps"][starttime]/1000)}')
                startdiff_seq.append([max(EPS, (datetime.datetime.fromtimestamp(ts/1000) - starttime).total_seconds()) for ts in currtimeseq[1:]])

                if self.mode != 'train':
                    label_seq.append(labels[sample_idx])

            # Save the cache
            cache_path = cache_path.joinpath(self.mode)
            cache_path.mkdir(parents=True, exist_ok=True)

            with open(f'{cache_path}/data_seq{str(int(self.seq_len))}.pickle', 'wb') as fp:
                pickle.dump(data_seq, fp)
            with open(f'{cache_path}/diff_seq{str(int(self.seq_len))}.pickle', 'wb') as fp:
                pickle.dump(diff_seq, fp)
            with open(f'{cache_path}/startdiff_seq{str(int(self.seq_len))}.pickle', 'wb') as fp:
                pickle.dump(startdiff_seq, fp)

            if self.mode != 'train':
                with open(f'{cache_path}/labels{str(int(self.seq_len))}.pickle', 'wb') as fp:
                    pickle.dump(label_seq, fp)

            self.logger.info(f'Cache generated for mode {self.mode}')
        else:
            try:
                # Load the cache
                cache_path = cache_path.joinpath(self.mode)

                with open(f'{cache_path}/data_seq{str(int(self.seq_len))}.pickle', 'rb') as fp:
                    data_seq = pickle.load(fp)
                with open(f'{cache_path}/diff_seq{str(int(self.seq_len))}.pickle', 'rb') as fp:
                    diff_seq = pickle.load(fp)
                with open(f'{cache_path}/startdiff_seq{str(int(self.seq_len))}.pickle', 'rb') as fp:
                    startdiff_seq = pickle.load(fp)

                if self.mode != 'train':
                    with open(f'{cache_path}/labels{str(int(self.seq_len))}.pickle', 'rb') as fp:
                        label_seq = pickle.load(fp)
                else:
                    label_seq = None

                self.logger.info(f'Cache loaded for mode {self.mode}')
            except FileNotFoundError:
                raise FileNotFoundError('You need to generate the cache first')

        # Reduce the number of samples for development mode
        if self.config['debug']:
            n = 1000
            data_seq = data_seq[:n]
            diff_seq = diff_seq[:n]
            startdiff_seq = startdiff_seq[:n]
            label_seq = label_seq if label_seq is None else label_seq[:n]

        return data_seq, diff_seq, startdiff_seq, label_seq
    
    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        channels = 1

        Tensor = torch.LongTensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        files = self.data_seq[idx]
        feature_diff = torch.tensor(self.diff_seq[idx][:-1]).reshape(-1, 1)
        feature_startdiff = torch.tensor(self.startdiff_seq[idx][:-1]).reshape(-1, 1)

        target_diff = torch.tensor([self.diff_seq[idx][-1]])
        target_startdiff = torch.tensor([self.startdiff_seq[idx][-1]])

        feature_data = []
        target_data = None

        for i, file in enumerate(files):
            with open(file, 'rb') as fp:
                data = pickle.load(fp)
                image = data['image']
                setting = data['setting']

            image = image.astype(np.float32)
            image = np.repeat(image[:, :, np.newaxis], channels, axis=2)

            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            if i == len(files) - 1:
                target_data = image
            else:
                feature_data.append(image.unsqueeze(0))

        assert 0 <= target_data.min() and target_data.max() <= 1, f"Problem with normalization in mode {self.mode}. Target image min: {target_data.min()} max: {target_data.max()}"


        if self.mode == "train":
            return torch.cat(feature_data, dim=0), feature_diff, feature_startdiff,\
                target_data, target_diff, target_startdiff, Tensor([int(files[-1].stem)])
        else:
            return torch.cat(feature_data, dim=0), feature_diff, feature_startdiff, \
                target_data, target_diff, target_startdiff, Tensor([int(files[-1].stem)]), Tensor([int(self.label_seq[idx])]), setting