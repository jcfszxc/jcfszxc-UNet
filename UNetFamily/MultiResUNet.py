#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 16:33
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : MultiResUNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class MultiResUNet(torch.nn.Module):
    """
    MultiResUNet

    Arguments:
        input_channels {int} -- number of channels in image
        num_classes {int} -- number of segmentation classes
        alpha {float} -- alpha hyperparameter (default: 1.67)

    Returns:
        [keras model] -- MultiResUNet model
    """

    def __init__(self, input_channels=3, num_classes=1, alpha=1.67):
        super().__init__()

        self.alpha = alpha
        self.n_channels = input_channels
        self.n_classes = num_classes

        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels, 32)
        self.in_filters1 = (
            int(32 * self.alpha * 0.167)
            + int(32 * self.alpha * 0.333)
            + int(32 * self.alpha * 0.5)
        )
        self.pool1 = torch.nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
        self.in_filters2 = (
            int(32 * 2 * self.alpha * 0.167)
            + int(32 * 2 * self.alpha * 0.333)
            + int(32 * 2 * self.alpha * 0.5)
        )
        self.pool2 = torch.nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)

        self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
        self.in_filters3 = (
            int(32 * 4 * self.alpha * 0.167)
            + int(32 * 4 * self.alpha * 0.333)
            + int(32 * 4 * self.alpha * 0.5)
        )
        self.pool3 = torch.nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
        self.in_filters4 = (
            int(32 * 8 * self.alpha * 0.167)
            + int(32 * 8 * self.alpha * 0.333)
            + int(32 * 8 * self.alpha * 0.5)
        )
        self.pool4 = torch.nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)

        self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
        self.in_filters5 = (
            int(32 * 16 * self.alpha * 0.167)
            + int(32 * 16 * self.alpha * 0.333)
            + int(32 * 16 * self.alpha * 0.5)
        )

        # Decoder path
        self.upsample6 = torch.nn.ConvTranspose2d(
            self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.concat_filters1 = 32 * 8 * 2
        self.multiresblock6 = Multiresblock(self.concat_filters1, 32 * 8)
        self.in_filters6 = (
            int(32 * 8 * self.alpha * 0.167)
            + int(32 * 8 * self.alpha * 0.333)
            + int(32 * 8 * self.alpha * 0.5)
        )

        self.upsample7 = torch.nn.ConvTranspose2d(
            self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2)
        )
        self.concat_filters2 = 32 * 4 * 2
        self.multiresblock7 = Multiresblock(self.concat_filters2, 32 * 4)
        self.in_filters7 = (
            int(32 * 4 * self.alpha * 0.167)
            + int(32 * 4 * self.alpha * 0.333)
            + int(32 * 4 * self.alpha * 0.5)
        )

        self.upsample8 = torch.nn.ConvTranspose2d(
            self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2)
        )
        self.concat_filters3 = 32 * 2 * 2
        self.multiresblock8 = Multiresblock(self.concat_filters3, 32 * 2)
        self.in_filters8 = (
            int(32 * 2 * self.alpha * 0.167)
            + int(32 * 2 * self.alpha * 0.333)
            + int(32 * 2 * self.alpha * 0.5)
        )

        self.upsample9 = torch.nn.ConvTranspose2d(
            self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2)
        )
        self.concat_filters4 = 32 * 2
        self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
        self.in_filters9 = (
            int(32 * self.alpha * 0.167)
            + int(32 * self.alpha * 0.333)
            + int(32 * self.alpha * 0.5)
        )

        self.conv_final = Conv2d_batchnorm(
            self.in_filters9, num_classes, kernel_size=(1, 1), activation="None"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        out = self.conv_final(x_multires9)

        return out
