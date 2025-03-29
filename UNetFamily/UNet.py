#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/27 16:07
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : UNet.py
# @Description   :

"""Full assembly of the parts to form the complete network"""

from UNetFamily.utils.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial double convolution
        self.inc = DoubleConv(n_channels, 64)

        # Encoder path (downsampling)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder path (upsampling with transposed convolution)
        # Note: We removed the factor parameter since our Up class doesn't use bilinear anymore
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final convolution to map to number of output classes
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output layer
        logits = self.outc(x)
        return logits
