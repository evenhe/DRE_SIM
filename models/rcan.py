import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn

## Default Conv Layer
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
## MeanShift Layer

## ResidualBlock (RB)
class ResidualBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):        
        super(ResidualBlock, self).__init__()
        
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
            
        self.block = nn.Sequential(*modules_body)

    def forward(self, x):
        residual = self.block(x)
        return x + residual


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        z = self.conv_du(y)
        return x * z


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        return x + res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks, block):
        super(ResidualGroup, self).__init__()
        modules_body = []
        if block == "RCAB":
            modules_body = [
                RCAB(
                    conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)) \
                for _ in range(n_resblocks)]
        elif block == "RB":
            modules_body = [
                ResidualBlock(
                    conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)) \
                for _ in range(n_resblocks)]
                
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return x + res


## Encoder
class Encoder(nn.Module):
    def __init__(self, conv, n_colors, n_feats, kernel_size, 
                 n_resgroups, bn=False, act=False):
        super(Encoder, self).__init__()
        h = []
        h.append(conv(n_colors, n_feats, kernel_size))
        h.append(conv(n_feats,  n_feats, 3))
        if bn: h.append(nn.BatchNorm2d(n_feats))
        if act: h.append(act())
        b = [ResidualGroup(
                conv, n_feats, kernel_size=3, reduction=None, 
                act=nn.ReLU(True), n_resblocks=3, block="RB") \
            for _ in range(n_resgroups)]
            
        self.head = nn.Sequential(*h)
        self.body = nn.Sequential(*b)
        
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return x + res

## Decoder
class Decoder(nn.Module):
    def __init__(self, n_feats, bn=False, act=False):
        super(Decoder, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(n_feats, n_feats, 3, stride=1))
        m.append(nn.Conv2d(n_feats, n_feats, 3, padding=0))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act: m.append(act())
        self.net = nn.Sequential(*m)
    
    def forward(self, x):
        return self.net(x)

class Fuser(nn.Module):
    def __init__(self, conv, n_colors, n_feats, kernel_size, 
                 reduction, n_resblocks, n_resgroups):        
        super(Fuser, self).__init__()
        h = []
        h.append(conv(n_colors*n_feats, n_feats, kernel_size))
        h.append(conv(n_feats, n_feats, 3))

        b = [ResidualGroup(
                conv, n_feats, kernel_size=3, reduction=reduction,
                act=nn.ReLU(True), n_resblocks=n_resblocks, block="RCAB") \
            for _ in range(n_resgroups)]
            
        self.head = nn.Sequential(*h)
        self.body = nn.Sequential(*b)

    def forward(self, x):
        if type(x) is list:
            stacked_input = torch.cat(x, 1)
        elif type(x) is torch.Tensor:
            stacked_input = x
        else:
            raise ValueError("Input Wrong!")
        fuse = self.head(stacked_input)
        res = self.body(fuse)
        return fuse + res
    

## Encode-Fuse-Decode SIM Network (EFDSIM)
class EFDSIM(nn.Module):
    def __init__(self,  n_feats=64, conv=default_conv):
        
        super(EFDSIM, self).__init__()
        

        n_resblocks = 3
        kernel_size = 3
        reduction = 16
        #act = nn.ReLU(True)
        
        # define encoder
        self.encode = Encoder(conv, 1, n_feats, kernel_size, n_resgroups=2)
        
        # define fuser
        
        # self.fuseP = Fuser(conv, n_phase, n_feats, kernel_size=3, 
        #                   reduction=reduction, n_resblocks=n_resblocks, n_resgroups=6)
        # self.fuseT = Fuser(conv, n_theta, n_feats, kernel_size=3, 
        #                   reduction=reduction, n_resblocks=n_resblocks, n_resgroups=6)
        
        self.fuse = Fuser(conv,  kernel_size=3, n_colors=1, 
                          reduction=reduction, n_resblocks=n_resblocks, n_resgroups=5)
        
        # define decoder
        self.decode = nn.Sequential(*[
            Decoder(n_feats=n_feats),
            conv(n_feats, 1, 1)
             ])
        
        
        ## hook
        self.hook = False
        self.path_features = None

    def forward(self, x):
        batch_size, seq_len, heigth, width = x.shape
        
        x = x.permute(1, 0, 2, 3) # tensor (9, B, H, W)
        x = x.view(batch_size, 1, heigth, width) # tensor (3, 3, B, 1, H, W)
        x1=self.encode(x)
        features = self.fuse(x1)
        y = self.decode(features) # tensor (B, 1, H, W)

        return y

