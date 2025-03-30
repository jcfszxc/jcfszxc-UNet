#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 03:59
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : DenseUNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class DenseUNet(nn.Module):
    def __init__(self, in_chan=3, out_chan=1, filters=128, num_conv=4):

        super(DenseUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, filters, 1)
        self.d1 = Single_level_densenet(filters, num_conv)
        self.down1 = Down_sample()
        self.d2 = Single_level_densenet(filters, num_conv)
        self.down2 = Down_sample()
        self.d3 = Single_level_densenet(filters, num_conv)
        self.down3 = Down_sample()
        self.d4 = Single_level_densenet(filters, num_conv)
        self.down4 = Down_sample()
        self.bottom = Single_level_densenet(filters, num_conv)
        self.up4 = Upsample_n_Concat(filters)
        self.u4 = Single_level_densenet(filters, num_conv)
        self.up3 = Upsample_n_Concat(filters)
        self.u3 = Single_level_densenet(filters, num_conv)
        self.up2 = Upsample_n_Concat(filters)
        self.u2 = Single_level_densenet(filters, num_conv)
        self.up1 = Upsample_n_Concat(filters)
        self.u1 = Single_level_densenet(filters, num_conv)
        self.outconv = nn.Conv2d(filters, out_chan, 1)
        self.n_channels = in_chan
        self.n_classes = filters
        self.bilinear = False

    def forward(self, x):
        x = self.conv1(x)
        x, y1 = self.down1(self.d1(x))
        x, y2 = self.down1(self.d2(x))
        x, y3 = self.down1(self.d3(x))
        x, y4 = self.down1(self.d4(x))
        x = self.bottom(x)
        x = self.u4(self.up4(x, y4))
        x = self.u3(self.up3(x, y3))
        x = self.u2(self.up2(x, y2))
        x = self.u1(self.up1(x, y1))
        x1 = self.outconv(x)
        # x1 = F.softmax(x1, dim=1)
        # print(x1.shape)
        return x1
