from .skip import skip
from .resnet import ResNet
from .unet_model import UNet
from .rcan import EFDSIM
import torch.nn as nn
from .unet_model import UNet
#def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
def get_net(input_depth,NET_TYPE,n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode='bilinear',downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad='0', act_fun=act_fun)

    elif NET_TYPE == 'unet1':
        net=UNet()
    elif NET_TYPE == 'ResNet':
        net = ResNet(input_depth, act_fun='LeakyReLU')
    elif NET_TYPE == 'RCAN':
        net=EFDSIM()
    else:
        assert False

    return net