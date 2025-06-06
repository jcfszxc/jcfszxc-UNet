#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 03:29
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : BARUNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class BARUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        self.n_channels = img_ch
        self.n_classes = output_ch
        self.bilinear = False
        super(BARUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = BABasicBlock(ch_in=64, ch_out=128)
        self.Conv3 = BABasicBlock(ch_in=128, ch_out=256)
        self.Conv4 = BABasicBlock(ch_in=256, ch_out=512)
        self.Conv5 = BABasicBlock(ch_in=512, ch_out=1024)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path

        x1 = self.Conv1(x)
        x1 = self.cbam1(x1) + x1
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.cbam2(x2) + x2
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.cbam3(x3) + x3
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.cbam4(x4) + x4
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1, dim=1)  # mine
        return d1


# ===================================================================
