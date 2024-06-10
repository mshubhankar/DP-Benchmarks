import torch
from data_normalization import data_normalization, standardize
import torch.nn as nn
import numpy as np

class ScaledWSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        print("ScaledWSConv2d")
        nn.Conv2d.__init__(self, in_channels, out_channels,
                           kernel_size, stride,
                           padding, dilation,
                           groups, bias)
        if gain:
            self.gain = nn.Parameter(
                torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        # Epsilon, a small constant to avoid dividing by zero.
        self.eps = eps

    def get_weight(self):
        # Get Scaled WS weight OIHW;
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3],
                          keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3],
                        keepdims=True)
        weight = (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)


class LogisticRegresion(torch.nn.Module):
    def __init__(self, input_dim, output_dim, norm=None, norm_stats=None, num_groups=32):
        super(LogisticRegresion, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        if norm is None:
            self.norm = torch.nn.Identity()
        if norm == "GroupNorm":
            self.norm = torch.nn.GroupNorm(num_groups, input_dim, affine=False)
        if norm == "DataNorm":
            self.norm = lambda x: standardize(x, norm_stats)
    
    def forward(self, x):
        out = self.linear(self.norm(x))
        return out


class TwoLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_neurons = 256, norm=None, norm_stats=None, num_groups=32):
        super(TwoLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, intermediate_neurons)
        if norm is None:
            self.norm = torch.nn.Identity()
        if norm == "GroupNorm":
            self.norm = torch.nn.GroupNorm(num_groups, input_dim, affine=False)
        if norm == "DataNorm":
            self.norm = lambda x: standardize(x, norm_stats)
        
        self.linear2 = torch.nn.Linear(intermediate_neurons, output_dim)
    
    def forward(self, x):
        out = self.linear(self.norm(x))
        out = torch.nn.functional.tanh(out)
        out = self.linear2(out)
        return out
        


class CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, weight_standardization=False, **kwargs):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.weight_standardization = weight_standardization
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=8,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            

            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]

            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                if self.weight_standardization:
                    conv2d = ScaledWSConv2d(c, v, kernel_size=3, stride=1, padding=1)
                    print("ScaledWSConv2d")
                else:
                    conv1d = nn.Conv1d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv1d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 5))
        else:
            self.classifier = nn.Linear(c * 16 * 16, 5)  # 8, 8

    def forward(self, x):
        if self.in_channels != 3:
            x = x.reshape(-1, self.in_channels, 512)
            x = self.norm(x)  # 16, 16
        x = self.features(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        