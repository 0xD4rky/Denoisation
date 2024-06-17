import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import cv2
import os
import glob
import shutil

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader




image = Image.open('Train/high/2.png')

## CONVERTING IMAGE TO TENSOR

tensor_image = torch.tensor(np.array(image))
#print(tensor_image.shape) # 400x600x3


class SCPA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SCPA,self).__init__()
        self.conv1_branch1 = nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = 1,padding = 0)
        self.conv2_branch1 = nn.Conv2d(in_channels,out_channels,kernel_size = 3,stride = 1,padding = 1)
        self.sigmoid = nn.Sigmoid()
        self.conv1_branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3_branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4_branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv =    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        branch1 = self.conv1_branch1(x)
        branch1 = self.conv2_branch1(branch1)
        branch2 = self.conv1_branch2(x)
        branch2a = self.conv2_branch2(branch2)
        branch2a = self.sigmoid(branch2a)
        branch2b = self.conv3_branch2(branch2)
        branch2 = branch2a*branch2b
        branch2 = self.conv4_branch2(branch2)
    
        #combining branch 1 and branch 2
        output = branch2 + branch1
        #final convolutional layer and add it to the orignal input
        final_conv = self.final_conv(output)
        SCPA_output = final_conv + x
        return SCPA_output