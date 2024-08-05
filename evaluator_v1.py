import csv
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import torchio as tio
from IPython import embed
import csv
import torch.nn.functional as F
import os
from tqdm import tqdm
import shutil
import random
from time import time
from pathlib import Path
from utils.io_v1 import npy2nii
from data.utils_v1 import create_hmap_v6
from scipy.ndimage import binary_dilation
import pandas as pd

class DetectionEvaluator:
    def __init__(self, config) -> None:
        self._config = config
        self._patch_size = self._config['patch_size']
        self._overlap = self._config['overlap']
        self._confidence = self._config['confidence']

    def __call__(self, model, number_epoch, timesteamp, show=False):
        print(f'>start to Evaluation...{number_epoch}')
        test_names = []
        txt_paths = []
        if self._config['train_overfit']:
            file_paths = [
                self._config['train_overfit_path'],
            ]
        else:
            file_paths = [self._config['test_training_mode_name_path'],
                          self._config['test_testing_mode_name_path']]  
        if show:
            file_paths = [self._config['test_testing_mode_name_path']]

        for file_path in file_paths:
            test_names = read_names_from_csv(file_path)
            if 'training' in file_path:
                part = 'training'
                det_file = '_train'
            elif 'overfit' in file_path:
                part = 'training'
                det_file = '_overfit'
            else:
                part = 'testing'
                det_file = ''

            txt_path_ = f"/public_bme/data/xiongjl/lymph_det/runs/{self._config['model_name']}/bbox_txt/{timesteamp}_{number_epoch}{det_file}/"  

            txt_paths.append(txt_path_)
            model.eval()
            scale = [1., 1., 1.]
            step = [pahs - ovlap for pahs, ovlap in zip(self._patch_size, self._overlap)]
            pbar = tqdm(test_names)
            count = 0
            for name in pbar:
                filename =  f".../{part}/{name}.npy"  # the npy file path

                if Path(filename).exists():
                    pass
                else:
                    print(f'in the all whole {part} data, the name of {name} is not exist')
                    filename = f".../{part}/...npy"  
                
                image_data = np.load(filename)[0, :, :, :]
                mask_nii = tio.ScalarImage(f"{self._config['lymph_nodes_data_path']}{part}_mask/{name}/mediastinum.nii.gz")
                resample = tio.Resample(0.7)
                resample_nii = resample(mask_nii)
                mask_data = np.array(resample_nii.data.squeeze(0))
                dilated_mask = binary_dilation(mask_data, iterations=1)
                image_data, coords_crop = use_mask_and_crop(image_data, dilated_mask)
                shape = image_data.shape[:]
                image_patches, arr_pad_shape = sliding_window_3d_volume_padded(image_data, patch_size=self._patch_size, stride=step) 
                
                label_xyzwhd = name2coord(self._config, part, name)
                whole_hmap = np.zeros(arr_pad_shape)
                whole_pbm = np.zeros(arr_pad_shape)
                whole_hmap_two = np.zeros(arr_pad_shape)
                whole_whd = np.zeros(np.hstack(((3), arr_pad_shape)))
                whole_offset = np.zeros(np.hstack(((3), arr_pad_shape)))

                pred_bboxes = []
                for image_patch in image_patches:
                    with torch.no_grad():
                        image_input = image_patch['image'].unsqueeze(0)
                        point = image_patch['point'][1:]
                        order = image_patch['point'][0]
                        image_input = image_input.cuda()

                        pred_hmap, pred_whd, pred_offset = model(image_input)
                
                        whole_hmap = place_small_image_in_large_image(whole_hmap, pred_hmap.squeeze(0).squeeze(0).cpu(), point)
                        whole_whd = place_small_image_in_large_image(whole_whd, pred_whd.squeeze(0).cpu(), point)
                        whole_offset = place_small_image_in_large_image(whole_offset, pred_offset.squeeze(0).cpu(), point)

                whole_hmap = torch.from_numpy(whole_hmap[0: shape[0], 0: shape[1], 0: shape[2]]).unsqueeze(0).unsqueeze(0)
                whole_hmap_two = torch.from_numpy(whole_hmap_two[0: shape[0], 0: shape[1], 0: shape[2]]).unsqueeze(0).unsqueeze(0)
                whole_whd = torch.from_numpy(whole_whd[:, 0: shape[0], 0: shape[1], 0: shape[2]]).unsqueeze(0)
                whole_offset = torch.from_numpy(whole_offset[:, 0: shape[0], 0: shape[1], 0: shape[2]]).unsqueeze(0)
                image_data = torch.from_numpy(image_data[0: shape[0], 0: shape[1], 0: shape[2]]).unsqueeze(0)

                (x_min, x_max, y_min, y_max, z_min, z_max) = coords_crop[:]

                dilated_mask = dilated_mask[x_min:x_max, y_min:y_max, z_min:z_max]
                dilated_mask = torch.tensor(dilated_mask).unsqueeze(0).unsqueeze(0)

                if whole_hmap.shape == dilated_mask.shape:
                    whole_hmap = whole_hmap * dilated_mask
                else:
                    print(f'whole hmap shape : ({whole_hmap.shape}) != mask_data shape : ({dilated_mask.shape})')
        
                count += 1
                if name == '202004020032' or name == '202004260160':
                    npy2nii(whole_hmap, f'whole_hmap_{name}', dir_name=f'runs/{self._config["model_name"]}')
                    npy2nii(dilated_mask, f'whole_mask_{name}', dir_name=f'runs/{self._config["model_name"]}')
                    npy2nii(image_data, f'whole_image_{name}', dir_name=f'runs/{self._config["model_name"]}')
      
                pred_bboxes = decode_bbox(self._config, whole_hmap, whole_whd, whole_offset, scale, self._confidence, reduce=1., cuda=True, point=(0,0,0))
                pred_bboxes = pred_bboxes[0: 200]
   
                
                ground_truth_boxes = centerwhd_2nodes(label_xyzwhd, point=(0, 0, 0))
                pred_bboxes = nms_(pred_bboxes, thres=self._config['nms_threshold'])
                pred_bboxes = merge_boxes(pred_bboxes, threshold=0.25)  
               
                txt_path = f"{self._config['root_path']}/runs/{self._config['model_name']}/bbox_txt/{timesteamp}_{number_epoch}{det_file}/"  
                if not os.path.exists(txt_path):
                    os.makedirs(txt_path)
                for bbox in pred_bboxes:
                    hmap_score, x1, y1, z1, x2, y2, z2 = bbox
                    with open(f"{txt_path}/{name}.txt", 'a') as f:
                        x1 += x_min
                        x2 += x_min
                        y1 += y_min
                        y2 += y_min
                        z1 += z_min
                        z2 += z_min
                        f.write(f'nodule {hmap_score} {x1} {y1} {z1} {x2} {y2} {z2} {str(shape).replace(" ", "")}\n')

    
        return txt_paths


def create_array_from_boxes(box_list, shape, threshold=0.1):
    # 创建全零数组
    array = np.zeros(shape, dtype=int)
    
    for box in box_list:
        if len(box) == 7:
            score, x1, y1, z1, x2, y2, z2 = box
            if score > threshold:
                # 计算框的中心位置
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)
                cz = int((z1 + z2) // 2)
                
                # 设置框内所有位置的值为 1
                for i in range(max(0, int(x1)), min(shape[0], int(x2)+1)):
                    for j in range(max(0, int(y1)), min(shape[1], int(y2)+1)):
                        for k in range(max(0, int(z1)), min(shape[2], int(z2)+1)):
                            array[i, j, k] = 1
                # 设置中心位置周围的 3x3x3 区域为 2
                for i in range(max(0, int(cx-1)), min(shape[0], int(cx+2))):
                    for j in range(max(0, int(cy-1)), min(shape[1], int(cy+2))):
                        for k in range(max(0, int(cz-1)), min(shape[2], int(cz+2))):
                            array[i, j, k] = 2
        else:
            x, y, z, w, h, d = box
            x1 = x - w/2.
            y1 = y - h/2.
            z1 = z - d/2.
            x2 = x1 + w
            y2 = y1 + h
            z2 = z1 + d
            # x1, y1, z1, x2, y2, z2 = box
            # 设置框内所有位置的值为 1
            for i in range(max(0, int(x1)), min(shape[0], int(x2)+1)):
                for j in range(max(0, int(y1)), min(shape[1], int(y2)+1)):
                    for k in range(max(0, int(z1)), min(shape[2], int(z2)+1)):
                        array[i, j, k] = 1

    return array



def adjust_coordinates(coords_list, min_point):
    adjusted_coords = []
    min_x, min_y, min_z = min_point
    for coord in coords_list:
        x1, y1, z1, x2, y2, z2= coord
        adj_x1 = x1 - min_x
        adj_y1 = y1 - min_y
        adj_z1 = z1 - min_z
        adj_x2 = x2 - min_x
        adj_y2 = y2 - min_y
        adj_z2 = z2 - min_z

        w = adj_x2 - adj_x1
        h = adj_y2 - adj_y1
        d = adj_z2 - adj_z1
        xc = adj_x1 + w/2.
        yc = adj_y1 + h/2.
        zc = adj_z1 + d/2.
        if adj_x1 >= 0 and adj_y1 >= 0 and adj_z1 >= 0:
            adjusted_coords.append((int(xc), int(yc), int(zc), int(w), int(h), int(d)))
    return adjusted_coords



def use_mask_and_crop(image, mask):

    # 找到掩码的边界
    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    # 裁剪图像
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped_image, (x_min, x_max, y_min, y_max, z_min, z_max)


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


def number2name(number):
    df = pd.read_csv(".../csv_files/nnunet_data_output.csv")
    number_csv = int(number) + 970
    df_sec = df[df['number'] == number_csv]
    name = df_sec.iloc[0]['name']
    return name


def merge_boxes(boxes, threshold=0.4):
    # 将框按照分数从高到低排序
    boxes = sorted(boxes, key=lambda box: box[0], reverse=True)
    merged_boxes = []
    while boxes:
        # 取出分数最高的框
        max_score_box = boxes.pop(0)
        overlaps = []
        for i, box in enumerate(boxes):
            # 计算重叠部分的体积
            overlap = max(0, min(max_score_box[4], box[4]) - max(max_score_box[1], box[1])) * \
                      max(0, min(max_score_box[5], box[5]) - max(max_score_box[2], box[2])) * \
                      max(0, min(max_score_box[6], box[6]) - max(max_score_box[3], box[3]))
            # 计算小框的体积
            volume = min((box[4] - box[1]) * (box[5] - box[2]) * (box[6] - box[3]), \
                        (max_score_box[4] - max_score_box[1]) * (max_score_box[5] - max_score_box[2]) * (max_score_box[6] - max_score_box[3]))
            # 如果重叠部分占小框体积的比例大于阈值，则记录下来
            if overlap / volume > threshold:
                overlaps.append(i)
        # 如果有重叠的框，将它们与最大分数框合并
        if overlaps:
            for i in sorted(overlaps, reverse=True):
                overlap_box = boxes.pop(i)
                max_score_box = [max_score_box[0]] + \
                                [min(max_score_box[j+1], overlap_box[j+1]) for j in range(3)] + \
                                [max(max_score_box[j+4], overlap_box[j+4]) for j in range(3)]
        merged_boxes.append(max_score_box)

    return merged_boxes




def pool_nms(heat, kernel):
    pad = (kernel - 1) // 2
    if isinstance(heat, np.ndarray):
        heat = torch.from_numpy(heat)
    time_nn_func = time()
    if heat.device == 'cuda:0':
        pass
    else:
        heat = heat.cuda()
    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    heat = heat.cpu()
    hmax = hmax.cpu()
    keep = (hmax == heat).float()
    try:
        result = heat * keep
    except Exception as e:
        print(f"An error occurred: {e}")
    result = heat * keep
    return result


from scipy.ndimage import label, find_objects
def find_top_points_in_regions(mask, pred_hms, top_n=3):
    # 确保pred_hms没有多余的批处理维度
    if pred_hms.shape[0] == 1:
        pred_hms = pred_hms.squeeze(0)

    labeled_array, num_features = label(mask)
    slices = find_objects(labeled_array)
    top_points = []

    for slice_ in slices:
        region = pred_hms[slice_]
        # print(region.size)
        k = min(top_n, region.size) - 1
        # 使用np.argpartition找到最大的k个值的索引
        flat_indices = np.argpartition(-region.ravel(), k)[:k+1]
        # 转换回多维索引
        multi_dim_indices = np.unravel_index(flat_indices, region.shape)
        # 获取这些点的实际值
        values = region[multi_dim_indices]
        # 将这些点的索引和值存储起来
        for i, value in enumerate(values):
            # 调整索引以匹配原始pred_hms的维度，这里需要加上slice_的起始位置
            adjusted_indices = tuple(multi_dim_indices[dim][i] + slice_[dim].start for dim in range(len(slice_)))
            top_points.append((adjusted_indices, value))

    return top_points


def decode_bbox(config, pred_hms, pred_whds, pred_offsets, scale, confidence, reduce, point, cuda):
    pred_hms    = pool_nms(pred_hms, kernel = config['decode_box_kernel_size'])
    heat_map    = np.array(pred_hms[0, :, :, :, :])
    pred_whd    = pred_whds[0, :, :, :, :]
    pred_offset = pred_offsets[0, :, :, :, :]
    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(0) # .bool() # .squeeze(0).bool()
    mask_tensor = mask.bool()  # 转换为Tensor，如果已经是Tensor则不需要

    top_points = find_top_points_in_regions(mask_tensor.numpy(), heat_map)
    xyzwhds = []
    hmap_scores = []
    for point_info in top_points:
        coord, value = point_info
        x = coord[0]
        y = coord[1]
        z = coord[2]
        
        offset_x = pred_offset[0, x, y, z]
        offset_y = pred_offset[1, x, y, z]
        offset_z = pred_offset[2, x, y, z]
        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]

        center = ((x + offset_x) * reduce, (y + offset_y) * reduce, (z + offset_z) * reduce)
        center = [a / b for a, b in zip(center, scale)]
        xyzwhds.append([center[0], center[1], center[2], w, h, d])
        hmap_scores.append(value)
    predicted_boxes = centerwhd_2nodes(xyzwhds, point=point, hmap_scores=hmap_scores)

    return predicted_boxes


def pad_image(image, target_size):
    # 计算每个维度需要填充的数量
    padding = [(0, max(0, target_size - size)) for size in image.shape]
    # 使用pad函数进行填充
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # 返回填充后的图像
    return padded_image


def sliding_window_3d_volume_padded(arr, patch_size, stride, padding_value=0):
    """
    This function takes a 3D numpy array representing a 3D volume and returns a 4D array of patches extracted using a sliding window approach.
    The input array is padded to ensure that its dimensions are divisible by the patch size.
    :param arr: 3D numpy array representing a 3D volume
    :param patch_size: size of the cubic patches to be extracted
    :param stride: stride of the sliding window
    :param padding_value: value to use for padding
    :return: 4D numpy array of shape (num_patches, patch_size, patch_size, patch_size)
    """
    # regular the shape
    if len(arr.shape) != 3:
        arr = arr.squeeze(0)

    patch_size_x = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_z = patch_size[2]

    stride_x = stride[0]
    stride_y = stride[1]
    stride_z = stride[2]

    # Compute the padding size for each dimension
    pad_size_x = (patch_size_x - (arr.shape[0] % patch_size_x)) % patch_size_x
    pad_size_y = (patch_size_y - (arr.shape[1] % patch_size_y)) % patch_size_y
    pad_size_z = (patch_size_z - (arr.shape[2] % patch_size_z)) % patch_size_z

    # Pad the array
    arr_padded = np.pad(arr, ((0, pad_size_x), (0, pad_size_y), (0, pad_size_z)), mode='constant', constant_values=padding_value)

    # Extract patches using a sliding window approach
    patches = []
    order = 0
    for i in range(0, arr_padded.shape[0] - patch_size_x + 1, stride_x):
        for j in range(0, arr_padded.shape[1] - patch_size_y + 1, stride_y):
            for k in range(0, arr_padded.shape[2] - patch_size_z + 1, stride_z):
                patch = arr_padded[i:i + patch_size_x, j:j + patch_size_y, k:k + patch_size_z]
                if isinstance(patch, np.ndarray):
                    patch = torch.from_numpy(patch).unsqueeze(0)
                else:
                    patch = patch.unsqueeze(0)
                start_point = torch.tensor([order, i, j, k])
                add = {'image': patch, 'point': start_point}
                patches.append(add)
                order += 1
    # return np.array(patches)
    return patches, arr_padded.shape


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


def nms_(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    # print(f'dets is {dets}')
    x1 = [det[1] for det in dets]
    y1 = [det[2] for det in dets]
    z1 = [det[3] for det in dets]
    x2 = [det[4] for det in dets]
    y2 = [det[5] for det in dets]
    z2 = [det[6] for det in dets]
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) * (z2[i] - z1[i]) for i in range(len(x1))]
    scores = [det[0] for det in dets]
    order = order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # print(f'in the nms, the len of dets is {len(dets)}')
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = [max(x1[i], x1[j]) for j in order[1:]]
        xx2 = [min(x2[i], x2[j]) for j in order[1:]]
        yy1 = [max(y1[i], y1[j]) for j in order[1:]]
        yy2 = [min(y2[i], y2[j]) for j in order[1:]]
        zz1 = [max(z1[i], z1[j]) for j in order[1:]]
        zz2 = [min(z2[i], z2[j]) for j in order[1:]]

        w = [max(xx2[i] - xx1[i], 0.0) for i in range(len(xx1))]
        h = [max(yy2[i] - yy1[i], 0.0) for i in range(len(yy1))]
        d = [max(zz2[i] - zz1[i], 0.0) for i in range(len(zz1))]

        inters = [w[i] * h[i] * d[i] for i in range(len(w))]
        unis = [areas[i] + areas[j] - inters[k] for k, j in enumerate(order[1:])]
        ious = [inters[i] / unis[i] for i in range(len(inters))]

        inds = [i for i, val in enumerate(ious) if val <= thres]
         # return the rest boxxes whose iou<=thres

        order = [order[i + 1] for i in inds]

            # inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3
    result = [dets[i] for i in keep]
    # print(f'after the nms, the len of result is {len(result)}')
    return result



def name2coord(config, part, mhd_name):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyzwhd = []
    if config['data_type'] == 'masked':
        csv_file_dir = f".../csv_files/{part}_refine_crop.csv"
    else:
        csv_file_dir = f".../lymph_csv_refine/{part}_refine.csv"
    with open(csv_file_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            
            if row[0] == "'" + mhd_name:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                w = float(row[4]) 
                h = float(row[5]) 
                d = float(row[6]) 
                xyzwhd.append((x, y, z, w, h, d))
    return xyzwhd


def place_small_image_in_large_image(large_image, small_image, start_coords):

    if (start_coords[0] < 0 or start_coords[1] < 0 or start_coords[2] < 0 or
            start_coords[0] + small_image.shape[-3] > large_image.shape[-3] or
            start_coords[1] + small_image.shape[-2] > large_image.shape[-2] or
            start_coords[2] + small_image.shape[-1] > large_image.shape[-1]):
        raise ValueError("小图像的起始坐标超出大图像范围")
    
    # 获取小图像的坐标范围
    x_start, y_start, z_start = start_coords
    x_end = x_start + small_image.shape[-3]
    y_end = y_start + small_image.shape[-2]
    z_end = z_start + small_image.shape[-1]
    
    # 将小图像放入大图像中，选择最大值
    if len(large_image.shape) == 3:
        large_image[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )

    elif len(large_image.shape) == 4:
        large_image[:, x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[:, x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )
    else:
        print(f'large image shape should be 3 or 4, but now is {len(large_image.shape)}')
    return large_image


def centerwhd_2nodes(xyzwhds, point, hmap_scores=None):
    if hmap_scores != None:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd, hmap_score in zip(xyzwhds, hmap_scores):

            x, y, z, length, width, height = xyzwhd
            x1 = max(x - length/2.0, 0)
            y1 = max(y - width/2.0, 0)
            z1 = max(z - height/2.0, 0)
            x2 = x + length/2.0
            y2 = y + width/2.0
            z2 = z + height/2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([hmap_score, x1, y1, z1, x2, y2, z2])
        return result
    
    else:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd in xyzwhds:

            x, y, z, length, width, height = xyzwhd
            x1 = max(0, x - length / 2.0)
            y1 = max(0, y - width / 2.0)
            z1 = max(0, z - height / 2.0)
            x2 = x + length / 2.0
            y2 = y + width / 2.0
            z2 = z + height / 2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([x1, y1, z1, x2, y2, z2])

        return result
  

def normal_list(list):
    new_list = []
    for lit in list:
        if lit == []:
            continue
        else:
            for l in lit:
                new_list.append(l)
    return new_list




if __name__ == '__main__':
    
    from models.swin_unetr_nnunet_sdfnet import Merge_swinunetr_nnunet_model
    import torch
    import argparse
    from utils.io_v1 import get_config
    from plot.sample_2 import plot
    import datetime

    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    print('start eval')

    parser.add_argument("--config", type=str, default='.../config/lymph_nodes_det')
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)
    model = Merge_swinunetr_nnunet_model(config['patch_size'], in_channels=1)
    model = model.cuda()
    model_path = '.../runs/.......pt'
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    epoch = 135
    show = True
    evaluator = DetectionEvaluator(config)
    txt_paths = evaluator(model, epoch, timestamp, show) # generate the txt file
    metric_scores = plot(config, txt_paths, epoch, timestamp)

