import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculate attention weights
        att_weights = self.conv1(x)
        att_weights = self.sigmoid(att_weights)
        # Apply attention to input features
        x_att = x * att_weights
        return x_att
    
