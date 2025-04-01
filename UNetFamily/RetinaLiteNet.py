#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 17:42
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : RetinaLiteNet.py
# @Description   :


import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


# CBAM Block implementation
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# MultiHeadAttention implementation for PyTorch
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x shape: [batch, length, channels]
        out, _ = self.mha(x, x, x)
        return out


class TransFuseNet(nn.Module):
    def __init__(self, input_channels=3):
        super(TransFuseNet, self).__init__()

        self.n_channels = input_channels
        self.n_classes = 1

        # Encoder - Convolutional Blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
        )

        # Encoder - Transformer Block
        self.multihead_attention = MultiHeadSelfAttention(embed_dim=32, num_heads=4)

        # CBAM blocks
        self.cbam1 = CBAM(32)
        self.cbam2 = CBAM(32)
        self.cbam3 = CBAM(16)

        # Decoder - Upsampling Blocks
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
        )

        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1), nn.ReLU()
        )

        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1), nn.ReLU()
        )

        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Output layers
        self.output_BV = nn.Conv2d(8, 1, kernel_size=1)
        self.output_OD = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)

        # Transformer block
        batch_size, channels, height, width = conv3.shape
        transformer_input = conv3.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )
        transformer_output = self.multihead_attention(transformer_input)
        transformer_output = torch.mean(transformer_output, dim=1, keepdim=True)
        transformer_output = transformer_output.reshape(batch_size, 1, 1, channels)
        transformer_output = transformer_output.expand(
            batch_size, height, width, channels
        ).permute(0, 3, 1, 2)

        # Apply CBAM to transformer output
        att1 = self.cbam1(transformer_output)

        # Concatenate features
        fused_features = torch.cat([conv3, att1], dim=1)

        # Decoder
        decoder1 = self.decoder_block1(fused_features)
        att2 = self.cbam2(decoder1)
        decoder1 = torch.cat([att2, conv2], dim=1)
        decoder1 = self.decoder_conv1(decoder1)

        decoder2 = self.decoder_block2(decoder1)
        att3 = self.cbam3(decoder2)
        decoder2 = torch.cat([att3, conv1], dim=1)
        decoder2 = self.decoder_conv2(decoder2)

        decoder3 = self.decoder_block3(decoder2)

        # Outputs
        output_BV = torch.sigmoid(self.output_BV(decoder3))
        output_OD = torch.sigmoid(self.output_OD(decoder3))

        return output_BV


# Function to create the model
def create_transfuse_net(input_shape):
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 3
    return TransFuseNet(input_channels=input_channels)
