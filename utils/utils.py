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
