import os
from tkinter import X
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import scipy
import math
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
def fft2d(input, gamma=0.1):
   # temp = torch.permute_dimens(input, (0, 3, 1, 2))
   # temp=input
    #fft = .fft2d(tf.complex(temp, tf.zeros_like(temp)))
    fft=torch.fft.fft2(torch.complex(input,torch.zeros_like(input)))
    #absfft = tf.pow(tf.abs(fft)+1e-8, gamma)
    absfft=torch.pow(torch.abs(fft)+1e-8, gamma)
   # output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return torch.fft.fftshift(absfft)
def global_average_pooling2d(layer_in):
    return torch.mean(layer_in,(2,3),keepdim=True)
def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))
    return x * cdf


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    temp_x=x.detach().cpu().numpy().squeeze()
    temp_y=y.detach().cpu().numpy().squeeze()
    temp=compare_ssim(temp_y,temp_x)
   # print(temp)
    temp_ssim=torch.from_numpy(np.array(temp))
    ssim_loss = ssim_para * (1 - torch.mean(temp_ssim))
    mse_loss = mse_para * torch.mean(torch.square(y - x))

    return mse_loss + ssim_loss


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    mae_loss = mae_para * torch.mean(torch.abs(x-y))
    mse_loss = mse_para * torch.mean(torch.square(y - x))

    return mae_loss + mse_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

