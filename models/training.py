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

from main import *
from attention import *
from unet import *

import torch.optim as optim

train_low_dir  = 'data/Train/low'
train_high_dir = 'data/Train/high'
val_low_dir    = 'data/Val/low'
val_high_dir   = 'data/Val/high'

## creating datasets and dataloaders ##

train_dataset = LowLightDataset(train_low_dir, train_high_dir, transform=transform)
val_dataset = LowLightDataset(val_low_dir, val_high_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

def gaussian(window_size, sigma):
    gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)], dtype=torch.float32)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)

# Combined Loss Function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, output, target):
        # L1 loss
        l1_loss = self.l1_loss(output, target)
        
        # SSIM loss
        ssim_loss = self.ssim_loss(output, target)
        
        # Gradient loss
        grad_loss = self.gradient_loss(output, target)
        
        # Combined loss
        total_loss = 0.1 * ssim_loss + l1_loss + grad_loss
        
        return total_loss

    def gradient_loss(self, output, target):
        # Compute gradients
        output_grad_x = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        output_grad_y = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # Compute gradient loss
        grad_loss_x = F.l1_loss(output_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(output_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y


## MODEL LOADING ##
# Initialize the model, loss function, and optimizer
model = LowLightModel()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = CombinedLoss()

num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for low_img, high_img in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False):
        low_img, high_img = low_img.to(device), high_img.to(device)
        
        # Forward pass
        output = model(low_img)
        
        # Compute loss
        loss = criterion(output, high_img)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_psnr = 0.0
    with torch.no_grad():
        for low_img, high_img in val_loader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            output = model(low_img)
            val_psnr += psnr(output, high_img).item()
    
    train_loss /= len(train_loader)
    val_psnr /= len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, PSNR: {val_psnr:.2f} dB")
