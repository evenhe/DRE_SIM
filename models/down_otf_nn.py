#from typing_extensions import Self
import numpy as np
import torch
import torch.nn as nn 
import cv2
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import scipy as sci
def fft2d(input,input2, gamma=0.1):
    ###
    f=torch.fft.fft2(input)
    fshift=torch.fft.fftshift(f)
    abs1=torch.mul(fshift,input2)
    ifshift=torch.fft.ifftshift(abs1)
    ifft=torch.fft.ifft2(ifshift)
   # output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return ifft
def down_otf(img1,otf):
    #print(np.shape(img1))
    ans1=fft2d(img1,otf)
    #print(torch.max(out))
    return torch.real(ans1)
class down_otf_nn(nn.Module):
    def __init__(self):
        super(down_otf_nn, self).__init__()
        pool_2X=nn.AvgPool2d(2, stride=2)
    def forward(x,y):
        #pool_2X=nn.AvgPool2d(2, stride=2)
        #down=down_otf(x,y)
        #temp=pool_2X(down)
        return down_otf(x,y)
        