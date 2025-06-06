#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/28 17:12
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : dice_score.py
# @Description   :

import torch
from torch import Tensor


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    # Add clipping to prevent extreme values
    input = torch.clamp(input, min=0.0, max=1.0)

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # Increase epsilon for better numerical stability
    epsilon = 1e-5

    # Handle empty masks
    sets_sum = torch.where(sets_sum < epsilon, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-5,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # Add gradient clipping to prevent NaN
    input = torch.clamp(input, min=1e-7, max=1.0 - 1e-7)

    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
