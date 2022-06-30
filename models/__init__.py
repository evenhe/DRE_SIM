
import torch.nn as nn
from .unet_model import UNet
def get_net(NET_TYPE):
    if NET_TYPE == 'UNet':
        net=UNet()
    else:
        assert False
    return net