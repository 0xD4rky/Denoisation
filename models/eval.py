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
from training import *

import torch.optim as optim


model.eval()
transform = transforms.Compose([
    transforms.Resize((400, 600)),
    transforms.ToTensor() 
])

input_image_path = '/kaggle/input/dataset/augmented_Train/val/low/267.png'
input_image = Image.open(input_image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    enhanced_tensor = model(input_tensor)

    enhanced_tensor = enhanced_tensor.cpu()
enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze())
plt.imshow(enhanced_image)
plt.title('Enhanced Image')
plt.axis('off')
plt.show()

image = cv2.imread("img path")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off') 
plt.show()