import torch
import torchio 
import sys
import os
from IPython import embed
import torch.utils.data as data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.utils_v1 import *
import random
import csv
from torch.utils.data import DataLoader
import pandas as pd


def name2number(name):
    csv_path = '.../csv_files/nnunet_data_output.csv'
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['name'] == name:
                number = row['number']

    return number

def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    shuffle = False if split in ['testing', 'validation'] else config['shuffle']

    dataset = lymphDataset(config, split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], # collate_fn=collator
    )
    return dataloader


def get_center_from_csv(name):
    # 读取CSV文件
    df_det = pd.read_csv('.../csv_files/stage_one_hard_centers_loss.csv')
    df = pd.read_csv('.../csv_files/nnunet_fold_4_training_valida_testing_data_resample_07.csv')
    number = name2number(name)

    # 筛选与给定名称匹配的所有行
    filtered_df_det = df_det[df_det['name'] == name]
    filtered_df = df[(df['number'] == int(number)) & ((df['type'] == 'fn') | (df['type'] == 'fp'))]

    # print(f'filtered_df is {filtered_df}', flush=True)
    # 检查是否找到了匹配的行
    if not filtered_df.empty:
        # 从筛选后的DataFrame中随机选择一行
        random_row = filtered_df.sample(n=1)
        if not filtered_df_det.empty:
            random_row_from_df2 = filtered_df_det.sample(n=1)
            combined_rows = pd.concat([random_row, random_row_from_df2])
            random_row = combined_rows.sample(n=1)
        # 提取坐标
        x = random_row['x'].iloc[0]
        y = random_row['y'].iloc[0]
        z = random_row['z'].iloc[0]

        return x, y, z
    else:
        return (0, 0, 0)


class lymphDataset(data.Dataset):

    def __init__(self, 
                 config,
                 split,
                 ):

        super(lymphDataset, self).__init__()
        self._config = config
        self._split = split
        self._name_coords_dict_training = read_imgcoord_fromcsv_2dict(config)
        self.setup()

    def __getitem__(self, index):

        name = self.names[index]

        p = random.random()
        if p > 0.6 :
            if self._config['cascade']['cascade'] and p > 0.8:
                if self._split == 'training':
                    center = get_center_from_csv(name)
                else:
                    center = self.pop_coord(name)[0:3]  #* 选取绝对的正样本
            else:
                if self._config['train_overfit']:
                    center = (0, 0, 0)
                else:
                    center = self.pop_coord(name)[0:3]  #* 选取绝对的正样本
        else:
            center = (0, 0, 0)
        dct = random_crop_3d(self._config, self._split, name, crop_size=self._config['patch_size'], p=self._config['use_center_or_randomly_rate'], center=center, augmentatoin=False)
        return dct

    def __len__(self):

        return len(self.names)
        
    def setup(self):
        print('set up ')
        if self._config['debug_mode'] == True:
            file_path = f'.../csv_files/part_{self._split}_names.csv'
        elif self._config['train_overfit'] == True:
            file_path = self._config['train_overfit_path']
        else:
            file_path = f'.../csv_files/{self._split}_names.csv'

        self.names = read_names_from_csv(file_path)

        if self._split == 'training':
            # random.seed(1)
            random.shuffle(self.names)

    def update_item(self, name):
        if self._split == 'training':
            imgcoord_valid = pd.read_csv(f'.../csv_files/training_npyrefine.csv')
        elif self._split == 'validation':
            imgcoord_valid = pd.read_csv(f'.../csv_files/validation_npyrefine.csv')
        raw = imgcoord_valid[imgcoord_valid['name']=="'" + name]
        coords = []
        for i in range(len(raw)):
            x = raw.iloc[i, 1]
            y = raw.iloc[i, 2]
            z = raw.iloc[i, 3]
            width = raw.iloc[i, 4]
            height = raw.iloc[i, 5]
            depth = raw.iloc[i, 6]
            coords.append([x, y, z, width, height, depth]) # 这个是图像坐标系
        self._name_coords_dict_training[self._split][name] = coords

    def pop_coord(self, name):
        if len(self._name_coords_dict_training[self._split][name]) == 0:
            print(f'in the {self._split}, the {name}"s positive nodes is ok!!!!')
            self.update_item(name)
        selected_number = random.choice(self._name_coords_dict_training[self._split][name])
        self._name_coords_dict_training[self._split][name].remove(selected_number)
        return selected_number
    



class detCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']

    def __call__(self, batch):
        batch_images = []
        batch_hmap = []
        batch_mask = []
        batch_offset = []
        batch_whd = []
        batch_name = []
        batch_pbm = []
        for dct in batch:
            image = dct['input'] 
            hmap = dct['hmap']
            offset = dct['offset']
            mask = dct['mask']
            whd = dct['whd']
            name = dct['name']
    
            batch_images.append(image)
            batch_hmap.append(hmap)
            batch_mask.append(mask)
            batch_offset.append(offset)
            batch_whd.append(whd)
            batch_name.append(name)

        batch_images = [torch.from_numpy(np_array) for np_array in batch_images]

        dct = {}
        dct['hmap'] = torch.stack(batch_hmap).unsqueeze(1)
        dct['input'] = torch.stack(batch_images).unsqueeze(1)
        dct['mask'] = torch.stack(batch_mask)
        dct['offset'] = torch.stack(batch_offset)
        dct['whd'] = torch.stack(batch_whd)
        dct['name'] = batch_name
        return dct
