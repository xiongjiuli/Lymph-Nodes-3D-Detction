"""Module containing the trainer of the transoar project."""

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'
import torch
from tqdm import tqdm

from plot.sample_2 import plot
from utils.io_v1 import npy2nii
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from utils.logger import nnUNetLogger
from time import time

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, loss_info, 
        device, config, path_to_run, logger, timestamp, epoch, metric_start_val, scheduler, optimizer, evaluator
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config
        self._weight_decay = 3e-5
        self._writer = nnUNetLogger(loss_info)
        self._scaler = GradScaler()
        self._evaluator = evaluator
        self._main_metric_max_val = metric_start_val
        self._logger = logger
        self._timestamp = timestamp
        self._current_epoch = 0

    def _calculate_new_learning_rate(self):
        max_epochs = int(self._config['epochs'])
        initial_lr = float(self._config['lr'])
        exponent = 0.9
        return initial_lr * (1 - self._current_epoch / max_epochs)**exponent
    
    def _train_one_epoch(self):
        print(f'> Training {self._config["model_name"]}...{self._current_epoch}')
        self._model.train()
        total_steps = len(self._train_loader)
        pbar = tqdm(self._train_loader, miniters=max(1, total_steps // 10))

        train_loss_save = []
        r_Loss = 0
        whd_Loss = 0
        mse_los = 0
        save=True
        batch_loss = []

        for i, dct in enumerate(pbar):
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)
            pbm_target = dct['pbm'].to(device=self._device)

            self._optimizer.zero_grad()  # 清零梯度

            # Make prediction
            with autocast(enabled=False): # 这个是混合精度 autocast + GradScaler
                if image.shape[0] != 1 or (image.shape[1] != 1 and image.shape[1] != 3 and image.shape[1] != 2) \
                    or image.shape[2] != self._config['patch_size'][0]\
                    or image.shape[3] != self._config['patch_size'][1]\
                    or image.shape[4] != self._config['patch_size'][2]:
                    continue

                hmap_pred, whd_pred, offset_pred = self._model(image)

                loss, loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)

            if self._current_epoch % 1 == 0 and save and (hmap_target).max() > 0.9:
                npy2nii(pbm_target, f'{self._config["model_name"]}_train_pbm_target_{self._current_epoch}')
                npy2nii(image, f'{self._config["model_name"]}_training_input_{self._current_epoch}')
                npy2nii(hmap_target, f'{self._config["model_name"]}_training_hmap_target_{self._current_epoch}')
                npy2nii(hmap_pred, f'{self._config["model_name"]}_training_hmap_pred_{self._current_epoch}')
                save=False

            # 累积损失
            loss = loss / self._config['batch_size_true']
            self._scaler.scale(loss).backward()

            train_loss_save.append(loss.item())
            batch_loss.append(loss.item())
            r_Loss += loss_dict['offset'] / self._config['batch_size_true']
            whd_Loss += loss_dict['whd'] / self._config['batch_size_true']
            mse_los += loss_dict['mse'] / self._config['batch_size_true']

            # 每10个数据进行一次反向传播和参数更新
            if (i + 1) % self._config['batch_size_true'] == 0:
                # Backward pass
                max_norm = self._config['clip_max_norm']
                if max_norm > 0:
                    # self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)
                # Update model parameters
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                print(f'Epoch - {self._current_epoch} : - Batch_Loss - {np.mean(batch_loss)}')
                batch_loss = []


        if (i + 1) % self._config['batch_size_true'] != 0:
            # Backward pass
            max_norm = self._config['clip_max_norm']
            if max_norm > 0:
                # self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)
            # Update model parameters
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._optimizer.zero_grad()
            print(f'Epoch - {self._current_epoch} : - Batch_Loss - {np.mean(batch_loss)}')
            batch_loss = []

        #* 修改lr，根据epoch 
        new_lr = self._calculate_new_learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f'Epoch - {self._current_epoch} : train_Loss - {np.mean(train_loss_save)}')
        print(f"Epoch - {self._current_epoch} : {self._config['loss_coefs']['mse']} * mse_loss: {mse_los/len(train_loss_save)}")
        print(f"Epoch - {self._current_epoch} : r_Loss: {r_Loss/len(train_loss_save)}")
        print(f"Epoch - {self._current_epoch} : whd_Loss: {whd_Loss/len(train_loss_save)}")
        self._writer.log('train_losses', np.mean(train_loss_save), self._current_epoch)
        self._writer.log('lrs', new_lr, self._current_epoch)
        pbar.set_description(f"Training ")


    def _validate(self):
        print(f'> Validation {self._config["model_name"]}... {self._current_epoch}')
        self._model.eval()

        valid_loss_save = []
        hmap_Loss = 0
        r_Loss = 0
        whd_Loss = 0
        mse_loss_val = 0
        pbar = tqdm(self._val_loader)

        for i, dct in enumerate(pbar):
            # Put data to gpu
            image = dct['input'].to(device=self._device)
            hmap_target = dct['hmap'].to(device=self._device)
            whd_target = dct['whd'].to(device=self._device)
            offset_target = dct['offset'].to(device=self._device)
            mask_target = dct['mask'].to(device=self._device)
            pbm_target = dct['pbm'].to(device=self._device)

            if image.shape[0] != 1 or (image.shape[1] != 1 and image.shape[1] != 3 and image.shape[1] != 2) \
                or image.shape[2] != self._config['patch_size'][0]\
                or image.shape[3] != self._config['patch_size'][1]\
                or image.shape[4] != self._config['patch_size'][2]:
                continue
            with torch.no_grad():  # 确保在这个block中不计算梯度
                hmap_pred, whd_pred, offset_pred = self._model(image)

                loss, loss_dict = self._criterion(hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target)

                valid_loss_save.append(loss.item())
                r_Loss += loss_dict['offset']
                whd_Loss += loss_dict['whd']
                mse_loss_val += loss_dict['mse']

            pbar.set_description(f"Validation...")
        
        print(f"Epoch: {self._current_epoch} : valid_Loss: {np.mean(valid_loss_save)}")
        print(f"Epoch: {self._current_epoch} : {self._config['loss_coefs']['mse']} * mse_loss: {mse_loss_val/len(valid_loss_save)}")
        print(f"Epoch: {self._current_epoch} : valid_r_Loss: {r_Loss/len(valid_loss_save)}")
        print(f"Epoch: {self._current_epoch} : valid_whd_Loss: {whd_Loss/len(valid_loss_save)}")
        
        if self._current_epoch != 0:
            self._writer.log('val_losses', np.mean(valid_loss_save), self._current_epoch)

        if loss < self._config['best_loss']:
            txt_paths = self._evaluator(self._model, self._current_epoch, self._timestamp) # generate the txt file
            if len(txt_paths) == 0:   # txt_path is the whole path 
                print('txt_path is None...mean no nodes be detected, so will not do the plot function, and no image and txt will be saved')
            else:
                metric_scores = plot(self._config, txt_paths, self._current_epoch, self._timestamp)
                print(f'{metric_scores["recall_test"]*100.}')
                if metric_scores['recall_test'] >= self._main_metric_max_val: 
                    self._main_metric_max_val = metric_scores['recall_test']
                    loss_info_dict = self._writer.return_loss_info_dict()
                    self._save_checkpoint(
                        self._current_epoch,
                        f'{self._timestamp}_model_best_{self._main_metric_max_val}_{self._current_epoch}.pt',
                        loss_info_dict,
                    )
                if self._config['train_overfit']:
                    print('Epoch: %d, recall_test: %.4d', self._current_epoch, metric_scores["recall_test"])
                else:
                    print('Epoch: %d, recall_test: %.4d', self._current_epoch, metric_scores["recall_test"])
                    print('Epoch: %d, recall_train: %.4d', self._current_epoch, metric_scores["recall_train"])
                    self._writer.log('recall_test', metric_scores["recall_test"], self._current_epoch)
                    self._writer.log('recall_train', metric_scores["recall_train"], self._current_epoch)

    def run(self):
        print(f'Training -- epoch : {self._epoch_to_start}')
        if not self._config['train_overfit']:
            if self._epoch_to_start == 0:   # For initial performance estimation
                self._validate()
        for self._current_epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            time_epoch = time()
            self._train_one_epoch()

            if not self._config['train_overfit']:
                if self._current_epoch % self._config['val_interval'] == 0:
                    self._validate()

            self._scheduler.step()
            self._writer.log('epoch_end_timestamps', time()-time_epoch, self._current_epoch)
            self._writer.plot_progress_png(f'.../runs/{self._config["model_name"]}', self._timestamp)
            loss_info_dict = self._writer.return_loss_info_dict()
            self._save_checkpoint(self._current_epoch, f'{self._timestamp}_model_last.pt', loss_info_dict)

    def _save_checkpoint(self, num_epoch, name, loss_info_dict):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if f'{self._timestamp}_model_best' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'loss_info': loss_info_dict,
        }, self._path_to_run / name)
