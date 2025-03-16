import os
import sys
import logging
from utils.config import Config
from pathlib import Path
import torch
from dataset.csp_dataset import CSPDataset
from dataset.ae_csp_dataset import AECSPDataset
from optim.ae_trainer import AETrainer
from optim.forecast_trainer import ForecastTrainer


def run(cfg: Config, logger: logging.Logger):
    # Create output path
    cfg['output_path'] = Path(cfg['output_path']).joinpath(cfg['experiment'])
    if not Path.exists(Path(cfg['output_path'])):
        Path.mkdir(Path(cfg['output_path']), parents=True)

    # Device
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device is %s.' % cfg["device"])

    if cfg['debug']:
        cfg['epochs'] = 4

    # -- Pretraining --
    if cfg['pretrained_ae'] is None and cfg['train_forecast']:
        Path(cfg['output_path']).mkdir(parents=True, exist_ok=True)

        pretrain_dataset = AECSPDataset(cfg)

        aetrainer = AETrainer(cfg)
        model_path = aetrainer.train(pretrain_dataset)
        aetrainer.test(pretrain_dataset)

        cfg['pretrained_ae'] = model_path.as_posix()
        del pretrain_dataset, aetrainer

    # -- Forecasting --
    forecast_dataset = CSPDataset(cfg)
    
    forecast_trainer = ForecastTrainer(cfg)
    if cfg['train_forecast']:
        forecast_trainer.train(forecast_dataset)
    if cfg['test_forecast']:
        forecast_trainer.test(forecast_dataset)


def load_logger(cfg):
    # Load logger
    Path(cfg['logs_path']).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(cfg['logs_path'], f"log_{cfg['experiment']}")
    if os.path.exists(log_file):
        os.remove(log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.info(f"Settings : ")
    for setting in cfg.settings.keys():
        logger.info(f"{setting} : {cfg[setting]}")
    return logger


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main.py <config_filepath>')
        sys.exit(1)

    # Load configuration file
    config_filepath = sys.argv[1]
    assert Path(config_filepath).exists()

    cfg = Config()
    cfg.load_config(config_filepath)

    # Run the experiments
    cfg['experiment'] = '_'.join([
        cfg['difffeature'], f'indim{cfg["indim"]}', f'cdim{cfg["cdim"]}', f'zdim{cfg["zdim"]}', f'timedim{cfg["timedim"]}', 
        f'epochs{cfg["epochs"]}', f'patience{cfg["patience"]}', f'seq_len{cfg["seq_len"]}', f'n_flows{cfg["n_flows"]}', 
        f'n_blocks{cfg["n_blocks"]}', cfg['time_series_arch'], f'seed{cfg["seed"]}'
    ])

    logger = load_logger(cfg)
    run(cfg, logger)