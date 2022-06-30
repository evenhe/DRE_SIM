# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

def cat_(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))  
    x = torch.cat([x2, x1], dim=1)
    return x
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2,bias = True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
       # print(x1.shape)
       # print(x2.shape)
        #x = torch.cat([x2, x1], dim=1)
        x=cat_(x1,x2)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch,pixel_factor=2):
        super(outconv, self).__init__()
        self.up_conv=nn.Conv2d(in_ch,in_ch*(pixel_factor**2), 3,padding=1)
        self.pix_X2=nn.PixelShuffle(pixel_factor)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
       # x=self.up_conv(x)
        #x=self.pix_X2(x)
        x = self.conv(x)
        return x
