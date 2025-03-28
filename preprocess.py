#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/27 15:33
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 1.preprocess.py
# @Description   : 预处理眼底图像数据集并保存为单一文件（包括训练集和测试集）

import glob
import os
import numpy as np
from PIL import Image
import pickle
import h5py
import joblib

def preprocess_dataset(save_method='pickle', include_test=True):
    """
    预处理数据集并保存为单一文件
    
    参数:
    save_method: 保存方法，可选 'pickle', 'joblib' 或 'h5'
    include_test: 是否包含测试集
    """
    dataset_path = '../datasets/drive_eye/'
    train_path = dataset_path + 'training/'
    test_path = dataset_path + 'test/'
    
    # 创建输出目录
    output_dir = 'data/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集
    train_dataset = process_data_subset(train_path, subset_name="训练集")
    
    # 保存训练集
    train_output_file = save_data(train_dataset, output_dir, 'train_eye_dataset', save_method)
    train_info = {
        'num_samples': len(train_dataset['images']),
        'image_shape': train_dataset['images'].shape if len(train_dataset['images']) > 0 else None,
        'mask_shape': train_dataset['masks'].shape if len(train_dataset['masks']) > 0 else None,
        'label_shape': train_dataset['labels'].shape if len(train_dataset['labels']) > 0 else None,
        'output_file': train_output_file
    }
    
    test_info = None
    test_output_file = None
    
    # 处理测试集
    if include_test:
        test_dataset = process_data_subset(test_path, subset_name="测试集")
        
        # 保存测试集
        test_output_file = save_data(test_dataset, output_dir, 'test_eye_dataset', save_method)
        test_info = {
            'num_samples': len(test_dataset['images']),
            'image_shape': test_dataset['images'].shape if len(test_dataset['images']) > 0 else None,
            'mask_shape': test_dataset['masks'].shape if len(test_dataset['masks']) > 0 else None,
            'label_shape': test_dataset['labels'].shape if len(test_dataset['labels']) > 0 else None,
            'output_file': test_output_file
        }
    
    # 返回数据集信息
    return {
        'train': train_info,
        'test': test_info
    }

def process_data_subset(data_path, subset_name="数据集"):
    """
    处理数据子集（训练集或测试集）
    
    参数:
    data_path: 数据路径
    subset_name: 子集名称，用于打印
    """
    images_path = data_path + 'images/'
    mask_path = data_path + 'mask/'
    label_path = data_path + '1st_manual/'
    
    # 存储处理后的数据
    images = []
    masks = []
    labels = []
    filenames = []
    
    for image_path in glob.glob(images_path + '*.tif'):
        image_name = image_path.split('/')[-1]
        mask_name = image_name.split('.')[0] + '_mask.gif'
        label_name = image_name.split('.')[0].split('_')[0] + '_manual1.gif'
        mask_path_name = mask_path + mask_name
        label_path_name = label_path + label_name
        
        image_numpy = np.array(Image.open(image_path))
        mask_numpy = np.array(Image.open(mask_path_name))
        label_numpy = np.array(Image.open(label_path_name))

        image_numpy = np.array(image_numpy, dtype=np.float32) / 255.0
        mask_numpy = np.array(mask_numpy, dtype=np.float32) / 255.0
        label_numpy = np.array(label_numpy, dtype=np.float32) / 255.0
        
        print(f"处理{subset_name} {image_name}: 图像形状: {image_numpy.shape}, 掩码形状: {mask_numpy.shape}, 标签形状: {label_numpy.shape}")
        
        # 添加到列表
        images.append(image_numpy)
        masks.append(mask_numpy)
        labels.append(label_numpy)
        filenames.append(image_name)
    
    # 转换为numpy数组
    images_array = np.array(images)
    masks_array = np.array(masks)
    labels_array = np.array(labels)
    
    # 创建包含所有数据的字典
    dataset = {
        'images': images_array,
        'masks': masks_array,
        'labels': labels_array,
        'filenames': filenames
    }
    
    return dataset

def save_data(dataset, output_dir, file_prefix, save_method):
    """
    使用指定方法保存数据
    
    参数:
    dataset: 要保存的数据集
    output_dir: 输出目录
    file_prefix: 文件前缀
    save_method: 保存方法
    
    返回:
    输出文件路径
    """
    # 根据选择的方法保存数据
    if save_method == 'pickle':
        # 使用pickle保存为单一文件
        output_file = output_dir + file_prefix + '.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"使用pickle保存数据到: {output_file}")
        
    elif save_method == 'joblib':
        # 使用joblib保存 (对大型数组更高效)
        output_file = output_dir + file_prefix + '.joblib'
        joblib.dump(dataset, output_file, compress=3)
        print(f"使用joblib保存数据到: {output_file}")
        
    elif save_method == 'h5':
        # 使用HDF5格式保存
        output_file = output_dir + file_prefix + '.h5'
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('images', data=dataset['images'])
            f.create_dataset('masks', data=dataset['masks'])
            f.create_dataset('labels', data=dataset['labels'])
            # HDF5不直接支持Python字符串列表，需要转换
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('filenames', data=np.array(dataset['filenames'], dtype=dt))
        print(f"使用HDF5保存数据到: {output_file}")
    
    else:
        raise ValueError("不支持的保存方法。请选择 'pickle', 'joblib' 或 'h5'")
    
    print(f"保存完成! 保存了 {len(dataset['images'])} 张图像。")
    
    return output_file

def load_preprocessed_data(file_path, load_method=None):
    """
    加载预处理的数据
    
    参数:
    file_path: 数据文件路径
    load_method: 加载方法，如果为None，将根据文件扩展名自动选择
    """
    if load_method is None:
        # 根据文件扩展名确定加载方法
        if file_path.endswith('.pkl'):
            load_method = 'pickle'
        elif file_path.endswith('.joblib'):
            load_method = 'joblib'
        elif file_path.endswith('.h5'):
            load_method = 'h5'
        else:
            raise ValueError(f"无法从文件扩展名确定加载方法: {file_path}")
    
    if load_method == 'pickle':
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    elif load_method == 'joblib':
        dataset = joblib.load(file_path)
        return dataset
    
    elif load_method == 'h5':
        dataset = {}
        with h5py.File(file_path, 'r') as f:
            dataset['images'] = f['images'][:]
            dataset['masks'] = f['masks'][:]
            dataset['labels'] = f['labels'][:]
            dataset['filenames'] = list(f['filenames'][:])
        return dataset
    
    else:
        raise ValueError("不支持的加载方法。请选择 'pickle', 'joblib' 或 'h5'")

if __name__ == "__main__":
    # 可以选择保存方法: 'pickle', 'joblib' 或 'h5'
    save_method = 'h5'  # HDF5通常是处理图像数据的最佳选择
    dataset_info = preprocess_dataset(save_method=save_method, include_test=True)
    
    print("\n训练集信息:")
    for key, value in dataset_info['train'].items():
        print(f"{key}: {value}")
    
    if dataset_info['test']:
        print("\n测试集信息:")
        for key, value in dataset_info['test'].items():
            print(f"{key}: {value}")
    
    # 测试加载数据
    print("\n测试训练数据加载...")
    loaded_train_data = load_preprocessed_data(dataset_info['train']['output_file'])
    print(f"成功加载训练数据 - 图像数量: {len(loaded_train_data['images'])}")
    
    if dataset_info['test']:
        print("\n测试测试数据加载...")
        loaded_test_data = load_preprocessed_data(dataset_info['test']['output_file'])
        print(f"成功加载测试数据 - 图像数量: {len(loaded_test_data['images'])}")