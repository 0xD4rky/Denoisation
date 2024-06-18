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
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F




# image = Image.open('Train/high/2.png')
# ## CONVERTING IMAGE TO TENSOR
# tensor_image = torch.tensor(np.array(image))
# #print(tensor_image.shape) # 400x600x3


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
    
    
class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        xx = torch.arange(width).repeat(height, 1)
        yy = torch.arange(height).view(-1, 1).repeat(1, width)
        xx = xx.float() / (width - 1)
        yy = yy.float() / (height - 1)
        xx = xx.repeat(batch_size, 1, 1).unsqueeze(1)
        yy = yy.repeat(batch_size, 1, 1).unsqueeze(1)
        if x.is_cuda:
            xx = xx.cuda()
            yy = yy.cuda()
        # adding x and y cordinate to RGB channel
        x = torch.cat([x, xx, yy], dim=1)
        x = self.conv(x)
        return x   

class InvResBlock(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(InvResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size = 1,padding = 0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels,mid_channels,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels,out_channels,kernel_size = 1,padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        id = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += id
        x = self.relu(x)
        
        return x



from unet import *
from attention import AttentionBlock

## LETS DENIOSE OUR BRANCH ##

'''
convolutional block -> 4 inv residual block -> attention block -> convolution block 
'''
class DenoiseBranch(nn.Module):
    def __init__(self,in_channel=3,out_channel=3):
        super(DenoiseBranch,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channel,out_channels =256,kernel_size = 3,padding =1)
        self.inv_1 = InvResBlock(in_channels=256, mid_channels=128, out_channels=256)
        self.inv_2 = InvResBlock(in_channels=256, mid_channels=128, out_channels=256)
        self.inv_3 = InvResBlock(in_channels=256, mid_channels=128, out_channels=256)
        self.inv_4 = InvResBlock(in_channels=256, mid_channels=128, out_channels=256)
        self.attention = AttentionBlock(in_channels = 256)
        self.conv_2 = nn.Conv2d(in_channels=256,out_channels = 3,kernel_size = 3,padding =1)
    
    def forward(self,x):
        x = self.conv_1(x)
        x = self.inv_1(x)
        x = self.inv_2(x)
        x = self.inv_3(x)
        x = self.inv_4(x)
        x = self.attention(x)
        x = self.conv_2(x)
        return x

class SCPA_Branch(nn.Module):
    def __init__(self,in_channel=3,out_channel=3):
        super(SCPA_Branch,self).__init__()
        self.coord_layer = CoordConv(in_channels =3,out_channels = 5)
        self.SCPA_1 = SCPA(in_channels = 5, out_channels =5)
        self.SCPA_2 = SCPA(in_channels = 5, out_channels =5)
        self.SCPA_3 = SCPA(in_channels = 5, out_channels =5)
        self.SCPA_4 = SCPA(in_channels = 5, out_channels =5)
        self.SCPA_5 = SCPA(in_channels = 5, out_channels =5)
        self.conv_layer =nn.Conv2d(in_channels=5,out_channels =3,kernel_size = 3,padding =1)
        
    def forward(self,x):
        x = self.coord_layer(x)
        x = self.SCPA_1(x)
        x = self.SCPA_2(x)
        x = self.SCPA_3(x)
        x = self.SCPA_4(x)
        x = self.SCPA_5(x)
        x = self.conv_layer(x)
        return x 
    
class LowLightModel(nn.Module):
    def __init__(self,in_channel=3,out_channel=3):
        super(LowLightModel,self).__init__()
        self.SCPA_branch = SCPA_Branch(in_channel=3,out_channel=3)
        self.Denoiser = DenoiseBranch(in_channel=3,out_channel=3)
        self.Unet = Unet()
        self.Conv = nn.Conv2d(in_channels=3,out_channels =3,kernel_size = 3,padding =1)
        
    def forward(self,x):
        # denoiser branch
        denoised = self.Denoiser(x)
        
        #SCPA branch 
        SCPA = self.SCPA_branch(x)
        Unet_input = SCPA + x
        Unet_output = self.Unet(Unet_input)
        # padding
        diff = Unet_input.size(3) - Unet_output.size(3)
        pad_left = diff // 2
        pad_right = diff - pad_left

        # Pad Unet_output
        Unet_output = F.pad(Unet_output, (pad_left, pad_right, 0, 0))
        
        output = self.Conv(Unet_output)
        final_output = output + denoised
        return final_output
    

## ACCESSING DATASETS ##

class LowLightDataset(Dataset):
    def __init__(self, low_img_dir, high_img_dir, transform=None):
        self.low_img_dir = low_img_dir
        self.high_img_dir = high_img_dir
        self.low_images = sorted(os.listdir(low_img_dir))
        self.high_images = sorted(os.listdir(high_img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_img_dir, self.low_images[idx])
        high_img_path = os.path.join(self.high_img_dir, self.high_images[idx])
        low_image = Image.open(low_img_path).convert("RGB")
        high_image = Image.open(high_img_path).convert("RGB")

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        return low_image, high_image

transform = transforms.Compose([
    transforms.ToTensor(),
])

## calculating psnr scores ##

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * torch.log10(pixel_max / torch.sqrt(mse))

