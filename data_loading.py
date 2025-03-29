#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/27 16:24
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 3.data_loading.py
# @Description   :


import os
import h5py
import random
import matplotlib.pyplot as plt


def load_preprocessed_data(file_path, load_method=None):
    """
    加载预处理的数据

    参数:
    file_path: 数据文件路径
    load_method: 加载方法，如果为None，将根据文件扩展名自动选择

    返回:
    包含图像、掩码、标签和文件名的字典
    """
    if load_method is None:
        # 根据文件扩展名确定加载方法
        if file_path.endswith(".pkl"):
            load_method = "pickle"
        elif file_path.endswith(".joblib"):
            load_method = "joblib"
        elif file_path.endswith(".h5"):
            load_method = "h5"
        else:
            raise ValueError(f"无法从文件扩展名确定加载方法: {file_path}")

    print(f"使用 {load_method} 方法加载数据文件: {file_path}")

    if load_method == "pickle":
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    elif load_method == "joblib":
        dataset = joblib.load(file_path)
        return dataset

    elif load_method == "h5":
        dataset = {}
        with h5py.File(file_path, "r") as f:
            dataset["images"] = f["images"][:]
            dataset["masks"] = f["masks"][:]
            dataset["labels"] = f["labels"][:]
            # 对于HDF5中的字符串，需要转换回Python字符串列表
            if isinstance(f["filenames"][0], bytes):
                dataset["filenames"] = [
                    filename.decode("utf-8") for filename in f["filenames"][:]
                ]
            else:
                dataset["filenames"] = list(f["filenames"][:])
        return dataset

    else:
        raise ValueError("不支持的加载方法。请选择 'pickle', 'joblib' 或 'h5'")


def display_dataset_info(dataset):
    """显示数据集基本信息"""
    print("\n数据集信息:")
    print(f"图像数量: {len(dataset['images'])}")
    print(f"图像形状: {dataset['images'][0].shape}")
    print(f"掩码形状: {dataset['masks'][0].shape}")
    print(f"标签形状: {dataset['labels'][0].shape}")

    # 打印每个样本的文件名
    print("\n样本文件名:")
    for i, filename in enumerate(dataset["filenames"]):
        print(f"样本 {i+1}: {filename}")


def visualize_samples(dataset, num_samples=3):
    """Visualize random samples"""
    if len(dataset["images"]) < num_samples:
        num_samples = len(dataset["images"])

    sample_indices = random.sample(range(len(dataset["images"])), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        # Original image
        axes[i, 0].imshow(dataset["images"][idx])
        axes[i, 0].set_title(f"Original Image: {dataset['filenames'][idx]}")
        axes[i, 0].axis("off")

        # Mask
        axes[i, 1].imshow(dataset["masks"][idx], cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        # Label (vessel segmentation)
        axes[i, 2].imshow(dataset["labels"][idx], cmap="gray")
        axes[i, 2].set_title("Vessel Label")
        axes[i, 2].axis("off")

    plt.tight_layout()

    # Save image
    output_dir = "visualizations/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "sample_visualization.png")
