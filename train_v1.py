"""Script for training the transoar project."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
# import monai
from trainer_v1 import Trainer
from data.data_v1 import get_loader
from utils.io_v1 import get_config, write_json, get_meta_data, creat_logging
from models.res101_v1 import CenterNet
from models.swin_unetr_v1 import SwinUNETR
from models.swin_unetr_nnunet_sdfnet import Merge_swinunetr_nnunet_model_sdf
from models.swin_unetr_nnunet_nonlf import Merge_swinunetr_nnunet_model
from criterion_v1 import Criterion
import datetime
from evaluator_v1 import DetectionEvaluator


def train(config, timestamp):

    log_file_path = f'{config["root_path"]}/logfile/{config["model_name"]}-{timestamp}.log'
    logger = creat_logging(log_file_path)
    device = config['device']
    train_loader = get_loader(config, 'training')
    if config['train_overfit']:
        val_loader = get_loader(config, 'training')
    else:
        val_loader = get_loader(config, 'validation')

    if config['model_name'] == 'det_path':
        model = SwinUNETR(img_size=config['patch_size'], in_channels=1, out_channels=7, feature_size=48)

    elif config['model_name'] == 'SDF_noGua':   
        model = Merge_swinunetr_nnunet_model(config['patch_size'], in_channels=1)
        config['centernet_point'] = True

    elif config['model_name'] == 'SDF_nonlf':   
        model = Merge_swinunetr_nnunet_model(config['patch_size'], in_channels=1)

    elif config['model_name'] == 'centernet':
        model = CenterNet('resnet101', 1)

    if config['model_name'] == 'SDF+':
        model = Merge_swinunetr_nnunet_model_sdf(config['patch_size'], in_channels=1)

    else:
        print(f'model name is wrong! now the model name is {config["model_name"]}')

    model = model.to(device=device)
    criterion = Criterion(config) # .to(device=device)

    optim = torch.optim.Adam(
        model.parameters(), lr=float(config['lr'])
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['scheduler_steps'], gamma=config['gamma'])
    evaluator = DetectionEvaluator(config)
    # 统计模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params}")
    
    # Load checkpoint if applicable
    if config['resume'] is not False:
        checkpoint = torch.load(Path(config['resume']))
        # Unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss_info = checkpoint['loss_info']
        metric_start_val = checkpoint['metric_max_val']

    else:
        epoch = 0
        metric_start_val = 0.
        loss_info = {"train_loss" : [],
                     "train_loss_epoch" : [],
                     "valid_loss" : [],
                     "valid_loss_epoch" : [],
                     "ap_01_test" : [],
                     "ap_01_test_epoch" : [],
                     "lr" : [],
                     "lr_epoch" : [],
                     "ap_01_train" : [],
                     "ap_01_train_epoch" : [],
                     "recall_test" : [],
                     "recall_test_epoch" : [],
                     "precision_test" : [],
                     "precision_test_epoch" : [],
                     "recall_train" : [],
                     "recall_train_epoch" : [],
                     "precision_train" : [],
                     "precision_train_epoch" : [],
                     "epoch_time" : [],
                     "epoch_time_epoch" : []}

    # Init logging
    path_to_run = Path(os.getcwd()) / 'runs' / config['model_name']
    path_to_run.mkdir(exist_ok=True)

    # Get meta data and write config to run
    config.update(get_meta_data())
    print(f"{path_to_run} / {timestamp}_config.json")
    write_json(config, path_to_run / f'{timestamp}_config.json')
        
    trainer = Trainer(
        train_loader, val_loader, model, criterion, loss_info, device, config, 
        path_to_run, logger, timestamp, epoch, metric_start_val, scheduler, optim, evaluator
    )
    trainer.run()



if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Add minimal amount of args (most args should be set in config files)
    parser.add_argument("--config", type=str, default='lymph_nodes_det')

    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)

    # To get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    torch.backends.cudnn.benchmark = False  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    train(config, args, timestamp)
