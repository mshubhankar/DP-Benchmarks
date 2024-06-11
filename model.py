import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import math

import numpy as np
from collections import OrderedDict


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
    
    def weight_standardization(self, weight, eps=1e-5):
        c_out, c_in, *kernel_shape = weight.shape
        weight = weight.view(c_out, -1)

        # Calculate mean and variance
        var, mean = torch.var_mean(weight, dim=1, keepdim=True)

        # Normalize
        weight = (weight - mean) / (torch.sqrt(var + eps))

        # Change back to original shape and return
        return weight.view(c_out, c_in, *kernel_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight_standardization(self.weight, eps=1e-5),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, stride, group_count, width, count, index):
        super(BasicBlock, self).__init__()
        self.count = count
        self.strides = stride

        layers = []

        for i in range(self.count):
            if i==0:
                layers.append((f'Block_{index}ACT_{i}_0', nn.ReLU()))
                layers.append((f'Block_{index}_norm_{i}_0', nn.GroupNorm(num_channels=in_planes, num_groups=group_count)))
                layers.append((f'Block_{index}Conv_{i}_0', Conv2dSame(in_channels=in_planes, out_channels=width, stride=stride, kernel_size=3)))

                layers.append((f'Block_{index}ACT_{i}_1', nn.ReLU()))
                layers.append((f'Block_{index}_norm_{i}_1', nn.GroupNorm(num_channels=width, num_groups=group_count)))
                layers.append((f'Block_{index}Conv_{i}_1', Conv2dSame(in_channels=width, out_channels=width, kernel_size=3, stride=(1, 1))))

            else:
                layers.append((f'Block_{index}ACT_{i}_0', nn.ReLU()))
                layers.append((f'Block_{index}_norm_{i}_0', nn.GroupNorm(num_channels=width, num_groups=group_count)))
                layers.append((f'Block_{index}Conv_{i}_0', Conv2dSame(in_channels=width, out_channels=width, kernel_size=3, stride=(1, 1))))
                
                layers.append((f'Block_{index}ACT_{i}_1', nn.ReLU()))
                layers.append((f'Block_{index}_norm_{i}_1', nn.GroupNorm(num_channels=width, num_groups=group_count)))
                layers.append((f'Block_{index}Conv_{i}_1', Conv2dSame(in_channels=width, out_channels=width, kernel_size=3, stride=(1, 1))))

        self.sequential = nn.Sequential(OrderedDict(layers))

    def forward(self, inputs):
        
        out = self.sequential(inputs)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, strides, width, count, groups, index):
        super(ResidualBlock, self).__init__()
        self.in_planes = in_planes
        self.strides = strides
        self.width = width
        self.count = count
        self.groups = groups
        
        layers = []
        layers.append((f'Block_{index}_skip_act', nn.ReLU()))
        layers.append((f'Block_{index}_skip_norm', nn.GroupNorm(num_channels=in_planes, num_groups=groups)))
        layers.append((f'Block_{index}_skip_conv', Conv2dSame(in_planes, width, 1, stride=strides)))

        self.block = BasicBlock(stride=strides, group_count=groups, in_planes=in_planes, count=count, width=width, index=index)

        self.sequential = nn.Sequential(OrderedDict(layers))


    def forward(self, inputs):

        out = self.block(inputs)
        skip = self.sequential(inputs)

        return out + skip

class WideResNet(nn.Module):
    def __init__(self, num_classes: int = 10, depth: int = 28, width: int = 10, dropout_rate: float = 0.0, use_skip_init: bool = False, use_skip_paths: bool = True, which_conv: str = 'Conv2D', which_norm: str = 'GroupNorm', activation: str = 'relu', groups: int = 16,  is_dp: bool = True, is_training: bool = False):
        
        super().__init__()
        self.num_output_classes = num_classes
        self.depth = depth
        self.width = width
        self.which_norm = which_norm
        self.use_skip_init = use_skip_init
        self.use_skip_paths = use_skip_paths
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.resnet_blocks = (depth - 4) // 6
        self.activation = nn.ReLU()

        self.First_conv = Conv2dSame(3, 16, 3)
        # Conv2dSame(in out kernel)

        self.res1 = ResidualBlock(in_planes=16, count = self.resnet_blocks, strides=(1, 1), width=16*self.width, groups=groups, index=1)
        self.res2 = ResidualBlock(in_planes=16*self.width, count = self.resnet_blocks, strides=(2, 2), width=32*self.width, groups=groups, index=2)
        self.res3 = ResidualBlock(in_planes=32*self.width, count = self.resnet_blocks, strides=(2, 2), width=64*self.width, groups=groups, index=3)
        self.Final_norm = nn.GroupNorm(num_groups=groups, num_channels=64*self.width)

        self.Softmax = torch.nn.Linear(64*self.width, num_classes, bias=True)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight = kaiming_normal_(self.Softmax.weight, mode='fan_in')
    
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):

        out = self.First_conv(inputs)

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)

        out = self.activation(out)
        out = self.Final_norm(out)

        out = torch.mean(out, dim=(2,3), dtype=torch.float32)  #Output shape: (64*width, )     **For WRN-28-10, (32, 640)**
        out = self.Softmax(out)

        if self.dropout_rate > 0.0:
            dropout_rate = self.dropout_rate if self.is_training else 0.0
            dropout = torch.nn.Dropout(p=dropout_rate)
            out = dropout(out)

        return out