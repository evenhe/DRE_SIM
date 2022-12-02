import torch
import torch.nn as nn
import torch.nn.functional as F

def tv_loss(pred, reduction="mean"):
    nc, nt, nh, nw = pred.shape

    dtype = pred.dtype
    device = pred.device
    kxx = torch.tensor([[[[0, 0, 0], [1, -1, 0], [0, 0, 0]]]], dtype=dtype, device=device)
    kyy = torch.tensor([[[[0, 1, 0], [0, -1, 0], [0, 0, 0]]]], dtype=dtype, device=device)

    dxx = F.conv2d(pred, kxx)
    dyy = F.conv2d(pred, kyy)

    res = torch.sqrt(dxx**2 + dyy**2)

    if reduction is "mean":
        loss = torch.mean(res)
    elif reduction is "sum":
        loss = torch.sum(res)
    else:
        raise ValueError("input parameter reduction is wrong!")

    return loss