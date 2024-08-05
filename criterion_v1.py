import torch 
from tqdm import tqdm 
import numpy as np
import os 
import torch.nn.functional as F 
from torch import nn


def contains_nan(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return torch.isnan(tensor).any().item()


def Mse_Loss(pred, target):
    pred = pred.float()
    target = target.float()
    mse_loss = nn.MSELoss(reduction='sum')
    loss = mse_loss(pred, target)

    return loss


def _reg_l1_loss(pred, target, mask):
    pred = pred.permute(0,2,3,4,1)
    target = target.permute(0,2,3,4,1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1, 3)
    
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class Criterion(nn.Module):
    def __init__(self, config) -> None:
        super(Criterion, self).__init__()
        self._config = config
    def forward(self, hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target):
        r_loss =  _reg_l1_loss(offset_pred, offset_target, mask_target)
        whd_loss = _reg_l1_loss(whd_pred, whd_target, mask_target)
        mse_loss = Mse_Loss(hmap_pred, hmap_target)

        loss = self._config['loss_coefs']['mse'] * mse_loss + \
               self._config['loss_coefs']['offset'] * r_loss + \
               self._config['loss_coefs']['whd'] * whd_loss  
        loss_dict = {}
        loss_dict['mse'] = self._config['loss_coefs']['mse'] * mse_loss
        loss_dict['whd'] = self._config['loss_coefs']['whd'] * whd_loss
        loss_dict['offset'] = self._config['loss_coefs']['offset'] * r_loss
        return loss, loss_dict
    


    