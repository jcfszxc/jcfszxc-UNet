#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 04:15
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : MCUNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class MCUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        bilinear: bool = True,
        base_c: int = 32,
    ):
        super(MCUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.cbam1 = CBAM(channel=32)
        self.cbam2 = CBAM(channel=64)
        self.cbam3 = CBAM(channel=128)
        self.cbam4 = CBAM(channel=256)
        factor = 2 if bilinear else 1
        self.down4 = InceptionA(base_c * 8)
        # self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up_v1(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_v1(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_v1(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_v1(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    # def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.cbam1(x1)
        x2 = self.down1(x1)
        x2 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        m1 = self.out_conv(x)
        return m1
