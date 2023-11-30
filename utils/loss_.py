import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def tv_loss(pred,reduction="mean"):
    nc, nt, nh, nw = pred.shape
    res_out=0
    loss = 0*pred.detach().clone()
    dtype = pred.dtype
    device = pred.device
    kxx = torch.tensor([[[[0, 0, 0], [1, -1, 0], [0, 0, 0]]]], dtype=dtype, device=device)
    kyy = torch.tensor([[[[0, 1, 0], [0, -1, 0], [0, 0, 0]]]], dtype=dtype, device=device)
    [_,c,_,_]=np.shape(pred)
   # print(c)
    for i in range(c):
        temp=pred[:,i,:,:]
        temp=temp[None,:,:,:]
        dxx = F.conv2d(temp, kxx,padding=1)
        dyy = F.conv2d(temp, kyy,padding=1)
        res = torch.sqrt(dxx**2 + dyy**2)
        loss[:,i,:,:]=res


    return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

def hessian_loss(pred, reduction="mean"):
    nc, nt, nh, nw = pred.shape

    loss = 0*pred.detach().clone()
    dtype = pred.dtype
    device = pred.device
    kxx = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], dtype=dtype, device=device)
    kyy = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=dtype, device=device)
    kxy = torch.tensor([[[[0, 0, 0], [0, 1, -1], [0, -1, 1]]]], dtype=dtype, device=device)
    [_,c,_,_]=np.shape(pred)


    for i in range(c):
        temp=pred[:,i,:,:]
        temp=temp[None,:,:,:]
        dxx = F.conv2d(temp, kxx,padding=1)
        dyy = F.conv2d(temp, kyy,padding=1)
        dxy = F.conv2d(temp, kxy,padding=1)

        res = torch.sqrt(dxx**2 + 2 * dxy**2 + dyy**2)
        loss[:,i,:,:]=res


    return loss
