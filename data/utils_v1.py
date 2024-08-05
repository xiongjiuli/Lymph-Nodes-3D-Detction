import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import csv
import torch
import random
import torch.nn.functional as F 
import torch.nn as nn
from scipy.ndimage import rotate
from time import time
from scipy.ndimage import zoom
from pathlib import Path


def read_imgcoord_fromcsv_2dict(config):
    dct = {
        'training':{},
        'validation':{}
    }
    #* 读取csv文件中的世界坐标
    if config['train_overfit']:
        names = read_names_from_csv('.../csv_files/train_overfit_names.csv')
    else:
        names = read_names_from_csv('.../csv_files/training_names.csv')
    imgcoord = pd.read_csv(f'.../csv_files/training_npyrefine.csv')
    for name in names:
        raw = imgcoord[imgcoord['name']=="'" + name]
        coords = []
        for i in range(len(raw)):
            x = raw.iloc[i, 1]
            y = raw.iloc[i, 2]
            z = raw.iloc[i, 3]
            width = raw.iloc[i, 4]
            height = raw.iloc[i, 5]
            depth = raw.iloc[i, 6]
            coords.append([x, y, z, width, height, depth]) # 这个是图像坐标系
        dct['training'][name] = coords
    if config['train_overfit']:
        names = read_names_from_csv('.../csv_files/train_overfit_names.csv')
        imgcoord = pd.read_csv(f'.../csv_files/training_npyrefine.csv')
    else:
        names = read_names_from_csv('.../csv_files/validation_names.csv')
        imgcoord = pd.read_csv(f'.../csv_files/validation_npyrefine.csv')
    for name in names:
        raw = imgcoord[imgcoord['name']=="'" + name]
        coords = []
        for i in range(len(raw)):
            x = raw.iloc[i, 1]
            y = raw.iloc[i, 2]
            z = raw.iloc[i, 3]
            width = raw.iloc[i, 4]
            height = raw.iloc[i, 5]
            depth = raw.iloc[i, 6]
            coords.append([x, y, z, width, height, depth]) # 这个是图像坐标系
        dct['validation'][name] = coords

    return dct


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


def read_imgcoord_fromcsv(config, name, part):
    #* 读取csv文件中的世界坐标
    if config['data_type'] == 'masked':
        imgcoord = pd.read_csv(f'.../csv_files/{part}_refine_crop.csv')
    else:
        imgcoord = pd.read_csv(f'.../csv_files/{part}_npyrefine.csv')
    raw = imgcoord[imgcoord['name']=="'" + name]
    coords = []
    for i in range(len(raw)):
        x = raw.iloc[i, 1]
        y = raw.iloc[i, 2]
        z = raw.iloc[i, 3]
        width = raw.iloc[i, 4]
        height = raw.iloc[i, 5]
        depth = raw.iloc[i, 6]
        coords.append([x, y, z, width, height, depth]) # 这个是图像坐标系

    return coords


def parse_list(string):
    # 将形如 "[517 517 295]" 的字符串转换为列表
    return list(map(int, string.strip("[]").split()))


def extract_data_from_csv(name, csv_path):
    bbox_list = []
    other_data = None

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['name'] == name:
                bbox_list.append(eval(row['bbox']))
                if other_data is None:
                    # 只需要从第一行中获取其他数据
                    other_data = {
                        'number': row['number'],
                        'origin': eval(row['origin']),
                        'old_shape': eval(row['old_shape']),
                        'new_shape': parse_list(row['new_shape']),
                        'old_spacing': eval(row['old_spacing'])
                    }

    return bbox_list, other_data



def random_crop_3d(config, part, name, crop_size, p, center, augmentatoin=False):
    if config['data_type'] == 'masked':
        print(f'!!!!!!!!!!!!! data type be the masked!!!!!!!!!!!!!!')
    if config['train_overfit']:
        filename = f"/.../lymph_nodes/npy_data_whole_new/training/{name}.npy"
    else:
        filename =  f"/.../lymph_nodes/npy_data_whole_new/{part}/{name}.npy"
    if Path(filename).exists():
        pass
    else:
        print(f'in the all whole {part} data, the name of {name} is not exist')
        filename = f"/.../lymph_nodes/npy_data_whole_new/{part}/202005210019.npy"   
    image = np.load(filename)[0, :, :, :]
    if config['channels'] == 3:
        csv_path = '/.../nnDet/csv_files/nnunet_data_output.csv'
        label_xyzwhd, other_data = extract_data_from_csv(name, csv_path)
        number = other_data['number']

        seg_path = f"/.../lymph_nodes/resample_07_mask_seg_npy_fordet/{name}_{part}_{number}_seg.npy"
        if os.path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            seg_nii = tio.ScalarImage(f"/.../temp/imageTr_10data_results_nnunet/lymph_{number}.nii.gz")
            ori_affine = seg_nii.affine
            file_path = f'/.../temp/imageTr_10data_results_nnunet/lymph_{number}.npz'
            pbm = np.load(file_path)['probabilities'][1:2, :, :, :]
            pbm = np.swapaxes(pbm, 1, 3)
            pbm_nii = tio.ScalarImage(tensor=torch.tensor(pbm), affine=ori_affine)
            resample = tio.Resample(0.7)
            resampled_seg = resample(pbm_nii)
            np.save(seg_path, np.array(resampled_seg.data))
            seg = np.array(resampled_seg.data)

        mask_path = f"/.../lymph_nodes/resample_07_mask_seg_npy_fordet/{name}_{part}_{number}_mask.npy"
        if os.path.exists(mask_path):
            mask_meds = np.load(mask_path)
        else:
            mask_nii = tio.ScalarImage(f"/.../lymph_nodes/{part}_mask/{name}/mediastinum.nii.gz")
            resample = tio.Resample(0.7)
            mask_nii = resample(mask_nii)
            np.save(mask_path, np.array(mask_nii.data))
            mask_meds = np.array(mask_nii.data)

    elif config['channels'] == 2:
        csv_path = '/.../nnDet/csv_files/nnunet_data_output.csv'
        label_xyzwhd, other_data = extract_data_from_csv(name, csv_path)
        number = other_data['number']

        seg_path = f"/.../lymph_nodes/resample_07_mask_seg_npy_fordet/{name}_{part}_{number}_seg.npy"
        if os.path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            '''
            # seg_nii = tio.ScalarImage(f"/.../nnUNet2/DataFrame/nnUNet_raw/Dataset501_lymph/imagesTr_seg_pbm_result_fold_4/lymph_{number}.nii.gz")
            # resample = tio.Resample(0.7)
            # resampled_seg = resample(seg_nii)
            # np.save(seg_path, np.array(resampled_seg.data))
            # seg = np.array(resampled_seg.data)
            '''
            seg_nii = tio.ScalarImage(f"/.../temp/imageTr_10data_results_nnunet/lymph_{number}.nii.gz")
            ori_affine = seg_nii.affine
            file_path = f'/.../temp/imageTr_10data_results_nnunet/lymph_{number}.npz'
            pbm = np.load(file_path)['probabilities'][1:2, :, :, :]
            pbm = np.swapaxes(pbm, 1, 3)
            pbm_nii = tio.ScalarImage(tensor=torch.tensor(pbm), affine=ori_affine)
            resample = tio.Resample(0.7)
            resampled_seg = resample(pbm_nii)
            np.save(seg_path, np.array(resampled_seg.data))
            seg = np.array(resampled_seg.data)

    # new_shape = (crop_size, crop_size, crop_size)
    origin_coords = read_imgcoord_fromcsv(config, name, part)
    width, height, depth = image.shape[:]

    crop_width, crop_height, crop_depth = crop_size
    
    # pad the image if it's smaller than the desired crop size
    pad_width = max(0, crop_width - width)
    pad_height = max(0, crop_height - height)
    pad_depth = max(0, crop_depth - depth)
    if pad_height > 0 or pad_width > 0 or pad_depth > 0:
        image = np.pad(image, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        if config['channels'] == 3:
            seg = np.pad(seg, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
            mask_meds = np.pad(mask_meds, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        elif config['channels'] == 2:
            seg = np.pad(seg, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        width, height, depth = image.shape[:]
    is_center = False
    # if random.random() < p:
        # 80% chance to have one or some points in the cropped image
    if np.any(center != (0, 0, 0)):
        x, y, z = center
        is_center = True
        x_sta = int(max(0, x - crop_width + 1))
        x_stop = int(min(x + 1, width - crop_width))
        y_sta = int(max(0, y - crop_height + 1))
        y_stop = int(min(y + 1, height - crop_height))
        z_sta = int(max(0, z - crop_depth + 1))
        z_stop = int(min(z + 1, depth - crop_depth))
        if x_sta > x_stop:
            x_sta = x_stop - 10
        if y_sta > y_stop:
            y_sta = y_stop - 10
        if z_sta > z_stop:
            z_sta = z_stop - 10
        x1 = random.randint(x_sta, x_stop)
        x2 = x1 + crop_width
        y1 = random.randint(y_sta, y_stop)
        y2 = y1 + crop_height
        z1 = random.randint(z_sta, z_stop)
        z2 = z1 + crop_depth

    else:
        # 20% chance to randomly crop the image
        x1 = random.randint(0, width - crop_width)
        x2 = x1 + crop_width
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        z1 = random.randint(0, depth - crop_depth)
        z2 = z1 + crop_depth
    if config['train_overfit']:
        x1 = max(0, int(width/2. - crop_width/2.))
        y1 = max(0, int(height/2. - crop_height/2.))
        z1 = max(0, int(depth/2. - crop_depth/2.))
        x2 = int(x1 + crop_width)
        y2 = int(y1 + crop_height)
        z2 = int(z1 + crop_depth)
    cropped_image = image[x1:x2, y1:y2, z1:z2]
    if config['channels'] == 3:
        cropped_seg = seg[:, x1:x2, y1:y2, z1:z2]
        cropped_mask_meds = mask_meds[:, x1:x2, y1:y2, z1:z2]
    elif config['channels'] == 2:
        cropped_seg = seg[:, x1:x2, y1:y2, z1:z2]
    cropped_points = [(x-x1,y-y1,z-z1,w,h,d) for (x,y,z,w,h,d) in origin_coords if x1 <= x < x2 and y1 <= y < y2 and z1 <= z < z2]

    if augmentatoin == True:
        if random.random() < 0.5:
            pass
        elif random.random() < 0.8:
            cropped_image, cropped_points = rotate_img(cropped_image, cropped_points, rotation_range=(-15, 15))
            cropped_points = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in origin_coords if 0 <= x <= cropped_image.shape[0] and 0 <= y <= cropped_image.shape[1] and 0 <= z <= cropped_image.shape[2]]
        else:
            cropped_image = add_noise(cropped_image)

    #* bulid the other label
    mask = create_mask(cropped_points, crop_size, reduce=1) # 0.0s no save is so fast
    whd = create_whd(coordinates=cropped_points, shape=crop_size, reduce=1)
    offset = create_offset(coordinates=cropped_points, shape=crop_size, reduce=1)
    if config['channels'] == 3:
        ex_image = np.expand_dims(cropped_image, axis=0)
        image_3 = np.concatenate((ex_image, cropped_seg, cropped_mask_meds), axis=0)
    if config['centernet_point']:
        hmap = create_pointmap(cropped_points, shape=crop_size)
    else:
        hmap = create_hmap_v6(cropped_points, shape=crop_size)
    # print(f'the max of the hmap is {hmap.max()}')
    hmap = torch.from_numpy(hmap)
    offset = torch.from_numpy(offset)
    mask = torch.from_numpy(mask)
    whd = torch.from_numpy(whd)

    dct = {}
    dct['hmap'] = hmap.unsqueeze(0)
    dct['offset'] = offset
    dct['mask'] = mask
    if config['channels'] == 3:
        dct['input'] = torch.tensor(image_3)
    elif config['channels'] == 2:
        dct['input'] = torch.tensor(cropped_image).unsqueeze(0)
        dct['pbm'] = torch.tensor(cropped_seg)
    else:
        dct['input'] = torch.tensor(cropped_image).unsqueeze(0)
    dct['new_coords'] = cropped_points
    dct['name'] = name
    dct['origin_coords'] = origin_coords
    dct['whd'] = whd
    dct['is_center'] = is_center

    return dct


def process_boxes(boxes, origin_whd, coord):
    result = []
    for i in range(len(boxes)):
        x, y, z = boxes[i]
        w, h, d = origin_whd[i]
        x -= coord[0]
        y -= coord[1]
        z -= coord[2]
        if (x - w/2 < 0 or y - h/2 < 0 or z - d/2 < 0) or (x + w/2 >= 128 or y + h/2 >= 128 or z + d/2 >= 128):
            continue
        result.append([x, y, z])
    return result


def crop_padding(image, start_point, size):
    # 计算裁剪区域的坐标范围
    x_min = start_point[0] - size[0] // 2
    x_max = start_point[0] + size[0] // 2
    y_min = start_point[1] - size[1] // 2
    y_max = start_point[1] + size[1] // 2
    z_min = start_point[2] - size[2] // 2
    z_max = start_point[2] + size[2] // 2
    
    # 计算需要填充的大小
    pad_x_min = max(0, -x_min)
    pad_x_max = max(0, x_max - image.shape[0])
    pad_y_min = max(0, -y_min)
    pad_y_max = max(0, y_max - image.shape[1])
    pad_z_min = max(0, -z_min)
    pad_z_max = max(0, z_max - image.shape[2])
    
    # 对图像进行填充
    padded_image = np.pad(image, ((pad_x_min, pad_x_max), (pad_y_min, pad_y_max), (pad_z_min, pad_z_max)), mode='constant', constant_values=0)
    
    # 裁剪图像
    cropped_image = padded_image[x_min+pad_x_min:x_max+pad_x_min, y_min+pad_y_min:y_max+pad_y_min, z_min+pad_z_min:z_max+pad_z_min]
    
    return cropped_image


def create_mask(coordinates, shape, reduce=4, save=False, name=''):
    
    arr = np.zeros(tuple(np.array(shape) // reduce)) 
    for coord in coordinates:
        x, y, z = coord[0: 3]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[int(x)][int(y)][int(z)] = 1
    if save:
        np.save('/.../det//npy_data//{}_mask.npy'.format(name), arr)
    
    return arr


def create_whd(coordinates, shape, reduce=4, save=False):
    
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for i in range(len(coordinates)):
        x, y, z, w, h, d = coordinates[i]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = w
        arr[1][int(x)][int(y)][int(z)] = h
        arr[2][int(x)][int(y)][int(z)] = d
    if save:
        np.save('array.npy', arr)
    
    return arr


def create_offset(coordinates, shape, reduce=4, save=False):
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for coord in coordinates:
        x, y, z = coord[0:3]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = x - int(x)
        arr[1][int(x)][int(y)][int(z)] = y - int(y)
        arr[2][int(x)][int(y)][int(z)] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr


# * load time is 0.09s
def create_hmap(coordinates, shape, reduce=4, save=None, hmap_dir=''): # 1.37s, if save :4.33s
    arr = np.zeros(tuple(np.array(shape) // reduce))
    for coord in coordinates:
        x, y, z = coord
        arr[int(x / reduce)][int(y / reduce)][int(z / reduce)] = 1
    arr = gaussian_filter(arr, sigma=3)
    if arr.max() == arr.min():
        if save != None:
            np.save(hmap_dir, arr)
        return arr
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        if save != None:
            np.save(hmap_dir, arr)
        return arr



def create_gaussian_kernel(whd):
    size = int(np.mean(whd))
    if size % 2 == 1:
        size = 2 * size + 1
    else:
        size = 2 * size + 1
    kernel = np.zeros((size, size, size))
    center = tuple(s // 2 for s in (size, size, size))
    kernel[center] = 1
    gassian_kernel = gaussian_filter(kernel, sigma=size//6)
    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min)
    return normalized_arr


def create_gaussian_kernel_v3(whd):
    size = int(np.mean(whd))
    if size % 2 == 1:
        size = 2 * size + 1
    else:
        size = 2 * size + 1
    kernel = np.zeros((size, size, size))
    center = tuple(s // 2 for s in (size, size, size))
    kernel[center] = 1
    if size // 6 <= 3:
        sigma = 3
    else:
        sigma = size // 6
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)
    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min)

    return normalized_arr



def create_hmap_v4(zuobiao, shape):
    arr = np.zeros(shape)
    for coords in zuobiao:
        coord = [int(x) for x in coords[0:3]]
        whd = [int(x) for x in coords[3:6]]
        kernel = create_gaussian_kernel_v4(whd)
        arr = place_gaussian(arr, kernel, coord)
    return arr



def create_hmap_v5(coordinates, shape):
    arr = np.zeros(shape)
    for coords in coordinates:
        coord = [int(x) for x in coords[0:3]]
        whd = [int(x) for x in coords[3:6]]
        kernel = create_gaussian_kernel_v5(whd)
        arr = place_gaussian(arr, kernel, coord)

    return arr

def create_hmap_v6(coordinates, shape):
    arr = np.zeros(shape)
    for coords in coordinates:
        coord = [int(x) for x in coords[0:3]]
        whd = [int(x) for x in coords[3:6]]
        kernel = create_gaussian_kernel_v6(whd)
        arr = place_gaussian(arr, kernel, coord)

    return arr


def create_pointmap(centers, shape):
    arr = np.zeros(shape, dtype=int)
    block_size = (3, 3, 3)

    for center in centers:
        center = tuple(int(round(c)) for c in center[0:3])

        # 计算小块的范围
        block_range = [slice(max(0, center[dim] - block_size[dim] // 2),
                             min(shape[dim], center[dim] + block_size[dim] // 2 + 1))
                       for dim in range(len(shape))]

        # 确保小块不越界
        if all(0 <= center[dim] < shape[dim] for dim in range(len(shape))):
            arr[tuple(block_range)] = 1
        # else:
        #     print(f'center {center} is out of the shape {shape}!!!!!!')

    return arr




def create_gaussian_kernel_v4(whd):
    size_max = int(np.max(whd))
    size_min = int(np.min(whd))
    size_mid = int(sorted(whd)[1])

    array_large = create_gaussian_base(size_max, 0.5)
    array_small = create_gaussian_base(size_min, 0.5)
    array_midum = create_gaussian_base(size_mid, 0.5)

    combined_kernel = combine_gaussian_kernels(array_large, array_small, array_midum)

    return combined_kernel


def create_gaussian_kernel_v5(whd):
    # 定义新的维度
    new_dims_w = int(whd[0])   # 新的长方体的维度
    new_dims_h = int(whd[1])
    new_dims_d = int(whd[2])
    size_max = int(np.max(whd))

    if new_dims_w % 2 == 0:
        new_dims_w += 1
    if new_dims_h % 2 == 0:
        new_dims_h += 1
    if new_dims_d % 2 == 0:
        new_dims_d += 1
    if size_max % 2 == 0:
        size_max += 1

    new_w = new_dims_w / size_max
    new_h = new_dims_h / size_max
    new_d = new_dims_d / size_max

    gaussian_kernel = create_gaussian_base(size_max, 0.3)
    # 使用scipy.ndimage.zoom函数来伸缩高斯核
    rescaled_kernel = zoom(gaussian_kernel, (new_w, new_h, new_d))
    rescaleded_kernel = add_dim_inarray(rescaled_kernel)

    return rescaleded_kernel


def create_gaussian_kernel_v6(whd):
    # 定义新的维度
    new_dims_w = int(whd[0])   # 新的长方体的维度
    new_dims_h = int(whd[1])
    new_dims_d = int(whd[2])
    size_max = int(np.max(whd))

    if new_dims_w % 2 == 0:
        new_dims_w += 1
    if new_dims_h % 2 == 0:
        new_dims_h += 1
    if new_dims_d % 2 == 0:
        new_dims_d += 1
    if size_max % 2 == 0:
        size_max += 1

    new_w = new_dims_w / size_max
    new_h = new_dims_h / size_max
    new_d = new_dims_d / size_max

    gaussian_kernel = create_gaussian_base(size_max, 0.01)
    # 使用scipy.ndimage.zoom函数来伸缩高斯核
    rescaled_kernel = zoom(gaussian_kernel, (new_w, new_h, new_d))
    rescaleded_kernel = add_dim_inarray(rescaled_kernel)

    return rescaleded_kernel


def add_dim_inarray(array):
    shape = np.shape(array)
    w, h, d = shape

    if w % 2 == 0:
        w += 1
        new_array = np.ones((w, h, d))
        new_array[0:int((w-1)/2 + 1), :, :] = array[0:int((w-1)/2 + 1), :, :]
        new_array[int((w-1)/2 + 1), :, :] = array[int((w-1)/2 ), :, :]
        new_array[int((w-1)/2 + 2) : w+1 , :, :] = array[int((w-1)/2 + 1) : w, :, :]
        array = new_array
    if h % 2 == 0:
        h += 1
        new_array = np.ones((w, h, d))
        new_array[:, 0:int((h-1)/2 + 1), :] = array[:, 0:int((h-1)/2 + 1), :]
        new_array[:, int((h-1)/2 + 1), :] = array[:, int((h-1)/2 ), :]
        new_array[:, int((h-1)/2 + 2) : w+1 , :] = array[:, int((h-1)/2 + 1) : h, :]
        array = new_array
    if d % 2 == 0:
        d += 1
        new_array = np.ones((w, h, d))
        new_array[:, :, 0:int((d-1)/2 + 1)] = array[:, :, 0:int((d-1)/2 + 1)]
        new_array[:, :, int((d-1)/2 + 1)] = array[:, :, int((d-1)/2 )]
        new_array[:, :, int((d-1)/2 + 2) : d+1 ] = array[:, :, int((d-1)/2 + 1) : d]
        array = new_array

    return array


def create_gaussian_base(size, threshold):

    if size <= 9:
        _size = 9
        half_dis = (_size + 1) / 2.
    else:
        _size = size
        if _size % 2 != 1:  # 如果size是偶数就变成奇数
            half_dis = _size / 2.
            _size = _size + 1
        else:
            half_dis = (_size + 1) / 2.

    if threshold == 0.5:
        sigma = np.sqrt(half_dis**2 / (2 * np.log(2)))
    elif threshold == 0.8:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(5) - np.log(4))))
    elif threshold == 0.3:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(10) - np.log(3))))
    elif threshold == 0.01:
        sigma = np.sqrt(half_dis**2 / (4 * (np.log(10))))
    else:
        print(f'when x = distance, the y wrong input, now the threshold is {threshold}')

    kernel = np.zeros((int(_size), int(_size), int(_size)))
    center = tuple(s // 2 for s in (int(_size), int(_size), int(_size)))
    kernel[center] = 1
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)

    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间
    # print(f'in the create_gaussian_base , the max is {normalized_arr.max()}, the min is {normalized_arr.min()}')
    return normalized_arr


def combine_gaussian_kernels(kernel_large, kernel_small, kernel_midum):
    center_large = np.array(kernel_large.shape) // 2
    small_shape = np.array(kernel_small.shape[0]) // 2
    midum_shape = np.array(kernel_midum.shape[0]) // 2

    kernel_large[center_large[0] - small_shape : center_large[0] + small_shape + 1, 
                 center_large[1] - small_shape : center_large[1] + small_shape + 1, 
                 center_large[2] - small_shape : center_large[2] + small_shape + 1, ] += kernel_small[:, :, :]
    
    kernel_large[center_large[0] - midum_shape : center_large[0] + midum_shape + 1, 
                 center_large[1] - midum_shape : center_large[1] + midum_shape + 1, 
                 center_large[2] - midum_shape : center_large[2] + midum_shape + 1, ] += kernel_midum[:, :, :]
    
    arr_min = kernel_large.min()
    arr_max = kernel_large.max()
    normalized_arr = (kernel_large - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间
    # print(f'in the combine_gaussian_kernels , the max is {normalized_arr.max()}, the min is {normalized_arr.min()}')
    return normalized_arr



def place_gaussian(arr, kernel, pos):
    x, y, z = pos
    kx, ky, kz = kernel.shape
    # 计算高斯核在数组中的位置
    x1, x2 = max(0, x-kx//2), min(arr.shape[0], x+kx//2+1)
    y1, y2 = max(0, y-ky//2), min(arr.shape[1], y+ky//2+1)
    z1, z2 = max(0, z-kz//2), min(arr.shape[2], z+kz//2+1)
    # 计算高斯核在自身中的位置
    kx1, kx2 = max(0, kx//2-x), min(kx, kx//2-x+arr.shape[0])
    ky1, ky2 = max(0, ky//2-y), min(ky, ky//2-y+arr.shape[1])
    kz1, kz2 = max(0, kz//2-z), min(kz, kz//2-z+arr.shape[2])
    arr[x1:x2,y1:y2,z1:z2] = np.maximum(arr[x1:x2,y1:y2,z1:z2], kernel[kx1:kx2,ky1:ky2,kz1:kz2])

    return arr



def rotate_coords(coordss, angle, center):
    rotated_coordss = []
    for coords in coordss:
        # 将coords转换为NumPy数组
        coords = np.array(coords)
        
        # 计算旋转矩阵
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        
        # 将坐标点平移到旋转中心
        coords -= center
        
        # 旋转坐标点
        rotated_coords = np.dot(coords, R.T)
        
        # 将坐标点平移回原来的位置
        rotated_coords += center

        rotated_coords.tolist()
        rotated_coordss.append(rotated_coords)
    
    return rotated_coordss


def rotate_img(image, coords, whd, rotation_range=(-15, 15)):
    # 将coords和whd转换为NumPy数组
    coords = np.array(coords)
    whd = np.array(whd)
    # 计算旋转角度
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    # 旋转图像
    rotated_image = rotate(image, angle, axes=(1, 0), reshape=False, mode='constant')
    # 规范化数据
    rotated_image = np.clip(rotated_image, 0, 1)
    # 计算旋转矩阵
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    center = [i/2 for i in image.shape]
    # 计算旋转后的坐标
    rotated_coords = rotate_coords(coords, angle, center)
    # 计算旋转后的whd
    rotated_whd = np.dot(whd, R.T)
    
    return rotated_image, rotated_coords, rotated_whd


def add_noise(img, std=(0, 0.05)):
    if isinstance(img, torch.Tensor):
        image = tio.ScalarImage(tensor=img.unsqueeze(0), type=tio.INTENSITY)
    else:
        image = tio.ScalarImage(tensor=torch.tensor(img).unsqueeze(0), type=tio.INTENSITY)
    transform = tio.RandomNoise(std=std)
    noisy_image = transform(image)
    result = np.array(noisy_image.data.squeeze(0))
    result = np.clip(result, 0, 1)
    return result


def npy2nii(name, image_npy, root_dir='/.../uii/nii_temp/', suffix='', resample=None, affine=''):
    # csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_athcoord.csv'
    csv_dir = os.path.join(root_dir, 'annotations_pathcoord.csv')
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    if resample != None:
        if affine == '':
            print("affine isn't be given")
    else:
        affine = image.affine
    
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4')



