#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 12:55
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : ResUNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class ResUNet(nn.Module):
    def __init__(self, channel=3, out_channels=1):
        super(ResUNet, self).__init__()
        self.n_channels = channel
        self.n_classes = out_channels
        self.bilinear = False

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(64, 128, 2, 1)
        self.residual_conv_2 = ResidualConv(128, 256, 2, 1)

        self.bridge = ResidualConv(256, 512, 2, 1)

        self.upsample_1 = Upsample(512, 512, 2, 2)
        self.up_residual_conv1 = ResidualConv(512 + 256, 256, 1, 1)

        self.upsample_2 = Upsample(256, 256, 2, 2)
        self.up_residual_conv2 = ResidualConv(128 + 256, 128, 1, 1)

        self.upsample_3 = Upsample(128, 128, 2, 2)
        self.up_residual_conv3 = ResidualConv(128 + 64, 64, 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, out_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
