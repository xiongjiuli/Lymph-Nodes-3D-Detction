"""Helper functions for input/output."""

import os
import json
import logging
import pickle
from pathlib import Path
import sys
import socket
import subprocess
import logging
import numpy as np
import torch
import yaml
import SimpleITK as sitk
import torchio as tio
PATH_TO_CONFIG = Path("./config/")


def get_config(config_name):
    """Loads a .yaml file from ./config corresponding to the name arg.

    Args:
        config_name: A string referring to the .yaml file to load.

    Returns:
        A container including the information of the referred .yaml file and information
        regarding the dataset, if specified in the referred .yaml file.
    """
    with open(PATH_TO_CONFIG / (config_name + '.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)

    return config

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        # print(f'file_path is {file_path}')
        # print(f'the data is {data}')
        json.dump(data, file, indent=3)
    # print(f'json file save in the {file_path}')


def get_meta_data():
    meta_data = {}
    # meta_data['git_commit_hash'] = subprocess.check_output(['git',  'ls-remote', 'git@github.com:xiongjiuli/lymph_det.git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    meta_data['python_version'] = sys.version.splitlines()[0]
    meta_data['gcc_version'] = sys.version.splitlines()[1]
    meta_data['pytorch_version'] = torch.__version__
    meta_data['host_name'] = socket.gethostname()

    return meta_data



def creat_logging(log_name):
# 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建一个handler，用于将日志写入到文件中
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 将handler添加到logger中
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger



def npy2nii(image, suffix='', dir_name='nii_temp'):
    image_npy = image
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if image_npy.requires_grad:
        image_npy = image_npy.detach()
    # print(image_npy.dtype)
    if image_npy.dtype == torch.float16:
        image_npy = image_npy.float()
    image_npy = image_npy.cpu()
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 1]])

    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save(f'.../{dir_name}/{suffix}.nii.gz')
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save(f'.../{dir_name}/{suffix}.nii.gz')
    elif len(image_npy.shape) == 5:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy[0, :, :, :, :], affine=affine)
        image_nii.save(f'.../{dir_name}/{suffix}.nii.gz')
    else: 
        print(f'in npy2nii...DIM ERROR : npy.dim != 3 or 4 or 5, the image_npy shape is {image_npy.shape}')



