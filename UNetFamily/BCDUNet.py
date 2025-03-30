#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 17:14
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : BCDUNet.py
# @Description   : 




import torch
import torch.nn as nn
from UNetFamily.utils.unet_parts import *


class BCDU_net_D3(nn.Module):
    def __init__(self, N=256, num_channels=3, num_classes=1):
        super(BCDU_net_D3, self).__init__()
        self.n_channels = num_channels
        self.n_classes = num_classes
        
        self.N = N
        in_channels = num_channels
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(128, 256)
        self.drop3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block
        # D1
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.drop4_1 = nn.Dropout(0.5)
        
        # D2
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_2_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2_2 = nn.ReLU(inplace=True)
        self.drop4_2 = nn.Dropout(0.5)
        
        # D3
        self.conv4_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3_2 = nn.ReLU(inplace=True)
        self.drop4_3 = nn.Dropout(0.5)
        
        # Decoder
        self.up6 = UpConv(512, 256)
        self.conv_lstm6 = ConvLSTM2D(256, 128, kernel_size=3, padding=1, go_backwards=True)
        self.conv6 = ConvBlock(128, 256)
        
        self.up7 = UpConv(256, 128)
        self.conv_lstm7 = ConvLSTM2D(128, 64, kernel_size=3, padding=1, go_backwards=True)
        self.conv7 = ConvBlock(64, 128)
        
        self.up8 = UpConv(128, 64)
        self.conv_lstm8 = ConvLSTM2D(64, 32, kernel_size=3, padding=1, go_backwards=True)
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv9 = nn.Conv2d(2, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        drop3 = self.drop3(conv3)
        pool3 = self.pool3(conv3)
        
        # Dense Block
        # D1
        conv4 = self.relu4(self.conv4(pool3))
        conv4_1 = self.relu4_1(self.conv4_1(conv4))
        drop4_1 = self.drop4_1(conv4_1)
        
        # D2
        conv4_2 = self.relu4_2(self.conv4_2(drop4_1))
        conv4_2 = self.relu4_2_2(self.conv4_2_2(conv4_2))
        drop4_2 = self.drop4_2(conv4_2)
        
        # D3
        merge_dense = torch.cat([drop4_2, drop4_1], dim=1)
        conv4_3 = self.relu4_3(self.conv4_3(merge_dense))
        conv4_3 = self.relu4_3_2(self.conv4_3_2(conv4_3))
        drop4_3 = self.drop4_3(conv4_3)
        
        # Decoder with LSTM
        up6 = self.up6(drop4_3)
        
        # Reshape for LSTM
        N = self.N
        x1 = drop3.unsqueeze(1)  # (batch, 1, 256, N/4, N/4)
        x2 = up6.unsqueeze(1)    # (batch, 1, 256, N/4, N/4)
        merge6 = torch.cat([x1, x2], dim=1)  # (batch, 2, 256, N/4, N/4)
        
        conv_lstm6 = self.conv_lstm6(merge6)
        conv6 = self.conv6(conv_lstm6)
        
        up7 = self.up7(conv6)
        
        x1 = conv2.unsqueeze(1)  # (batch, 1, 128, N/2, N/2)
        x2 = up7.unsqueeze(1)    # (batch, 1, 128, N/2, N/2)
        merge7 = torch.cat([x1, x2], dim=1)  # (batch, 2, 128, N/2, N/2)
        
        conv_lstm7 = self.conv_lstm7(merge7)
        conv7 = self.conv7(conv_lstm7)
        
        up8 = self.up8(conv7)
        
        x1 = conv1.unsqueeze(1)  # (batch, 1, 64, N, N)
        x2 = up8.unsqueeze(1)    # (batch, 1, 64, N, N)
        merge8 = torch.cat([x1, x2], dim=1)  # (batch, 2, 64, N, N)
        
        conv_lstm8 = self.conv_lstm8(merge8)
        conv8 = self.conv8(conv_lstm8)
        conv9 = self.sigmoid(self.conv9(conv8))
        
        return conv9

class BCDU_net_D1(nn.Module):
    def __init__(self, N=256, num_channels=3, num_classes=1):
        super(BCDU_net_D1, self).__init__()
        self.n_channels = num_channels
        self.n_classes = num_classes
        
        self.N = N
        in_channels = num_channels
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(128, 256)
        self.drop3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # D1 Block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.drop4_1 = nn.Dropout(0.5)
        
        # Decoder
        self.up6 = UpConv(512, 256)
        self.conv_lstm6 = ConvLSTM2D(256, 128, kernel_size=3, padding=1, go_backwards=True)
        self.conv6 = ConvBlock(128, 256)
        
        self.up7 = UpConv(256, 128)
        self.conv_lstm7 = ConvLSTM2D(128, 64, kernel_size=3, padding=1, go_backwards=True)
        self.conv7 = ConvBlock(64, 128)
        
        self.up8 = UpConv(128, 64)
        self.conv_lstm8 = ConvLSTM2D(64, 32, kernel_size=3, padding=1, go_backwards=True)
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv9 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        drop3 = self.drop3(conv3)
        pool3 = self.pool3(conv3)
        
        # D1 Block
        conv4 = self.relu4(self.conv4(pool3))
        conv4_1 = self.relu4_1(self.conv4_1(conv4))
        drop4_1 = self.drop4_1(conv4_1)
        
        # Decoder with LSTM
        up6 = self.up6(drop4_1)
        
        # Reshape for LSTM
        N = self.N
        x1 = drop3.unsqueeze(1)  # (batch, 1, 256, N/4, N/4)
        x2 = up6.unsqueeze(1)    # (batch, 1, 256, N/4, N/4)
        merge6 = torch.cat([x1, x2], dim=1)  # (batch, 2, 256, N/4, N/4)
        
        conv_lstm6 = self.conv_lstm6(merge6)
        conv6 = self.conv6(conv_lstm6)
        
        up7 = self.up7(conv6)
        
        x1 = conv2.unsqueeze(1)  # (batch, 1, 128, N/2, N/2)
        x2 = up7.unsqueeze(1)    # (batch, 1, 128, N/2, N/2)
        merge7 = torch.cat([x1, x2], dim=1)  # (batch, 2, 128, N/2, N/2)
        
        conv_lstm7 = self.conv_lstm7(merge7)
        conv7 = self.conv7(conv_lstm7)
        
        up8 = self.up8(conv7)
        
        x1 = conv1.unsqueeze(1)  # (batch, 1, 64, N, N)
        x2 = up8.unsqueeze(1)    # (batch, 1, 64, N, N)
        merge8 = torch.cat([x1, x2], dim=1)  # (batch, 2, 64, N, N)
        
        conv_lstm8 = self.conv_lstm8(merge8)
        conv8 = self.conv8(conv_lstm8)
        conv9 = self.sigmoid(self.conv9(conv8))
        
        return conv9

# Example usage
if __name__ == "__main__":
    # Define model with input shape, num_channels and num_classes
    model = BCDU_net_D3(input_size=(256, 256), num_channels=3, num_classes=1)
    
    # Create a random input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Use BCELoss for binary segmentation
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)