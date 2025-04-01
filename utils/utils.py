#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 00:52
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : utils.py
# @Description   :


import os
import torch
import random
import logging
import numpy as np
from PIL import Image


def set_seed(seed):
    """
    Set seed for reproducibility.

    Args:
        seed (int): Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")

def set_deterministic_mode(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def vis_numpy_img(imgs, save_path):
    """
    Visualize a numpy image and save it to a specified path.

    Args:
        img (numpy.ndarray): The numpy image to be visualized.
        save_path (str): The path to save the visualized image.

    Returns:
        None
    """
    
    c, w, h = imgs[0].shape
    numpy_list = []
    
    blank_numpy = np.zeros((c, 8, h))
    
    for img in imgs:
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        numpy_list.append(img)
        numpy_list.append(blank_numpy)
    img_array = np.array(np.concatenate(numpy_list, axis=1) * 255).astype(np.uint8).transpose(2, 1, 0)
    img_concat = Image.fromarray(img_array)
    img_concat.save(save_path)
    