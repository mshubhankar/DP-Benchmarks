import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np


def standardize(x, bn_stats, grad_sample_mode=None):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()

    del bn_mean, bn_var, view
    torch.cuda.empty_cache()

    return x


def clip_data(data, max_norm):
    norms = torch.norm(data.reshape(data.shape[0], -1), dim=-1)
    scale = (max_norm / norms).clamp(max=1.0)
    data *= scale.reshape(-1, 1, 1, 1)
    return data


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats):
        super(StandardizeLayer, self).__init__()
        self.bn_stats = bn_stats

    def forward(self, x):
        return standardize(x, self.bn_stats)


class ClipLayer(nn.Module):
    def __init__(self, max_norm):
        super(ClipLayer, self).__init__()
        self.max_norm = max_norm

    def forward(self, x):
        return clip_data(x, self.max_norm)

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
        # print("self.weight.shape", self.weight.shape)
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

class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, weight_standardization=False, grad_sample_mode=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.weight_standardization=weight_standardization
        self.grad_sample_mode = grad_sample_mode
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
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
                self.norm = lambda x: standardize(x, bn_stats, self.grad_sample_mode)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.weight_standardization:
                    conv2d = ScaledWSConv2d(c, v, kernel_size=3, stride=1, padding=1)
                    print("ScaledWSConv2d")
                else:
                    conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ScatterLinear(nn.Module):
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        print("ScatterLinear")
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        # x = self.clip(x)
        # x = x.reshape(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CheXpert_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None,weight_standardization=False, grad_sample_mode=None, **kwargs):
        super(CheXpert_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.weight_standardization = weight_standardization
        self.grad_sample_mode = grad_sample_mode
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        print("size", size)

        if self.in_channels == 3:
            print("Multiple CNN layers")
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']

            self.norm = nn.Identity()
        else:
            print("2 Layer CNN with in channels: ", self.in_channels)
            cfg = [64, 'M', 64]

            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats, self.grad_sample_mode)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.weight_standardization:
                    conv2d = ScaledWSConv2d(c, v, kernel_size=3, stride=1, padding=1)
                    print("ScaledWSConv2d")
                else:
                    conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 5))
        else:
            self.classifier = nn.Linear(c * 28 * 28, 5)  # 8, 8

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 56, 56))  # 16, 16
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EYEPACS_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, weight_standardization=False, grad_sample_mode="no_op", **kwargs):
        super(EYEPACS_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.weight_standardization = weight_standardization
        self.grad_sample_mode = grad_sample_mode
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            print("Multiple CNN layers with in channels", self.in_channels)
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            print("2 Layer CNN with in channels:", self.in_channels)
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]

            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats, self.grad_sample_mode)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.weight_standardization:
                    conv2d = ScaledWSConv2d(c, v, kernel_size=3, stride=1, padding=1)
                    print("ScaledWSConv2d")
                else:
                    conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 5))
        else:
            self.classifier = nn.Linear(c * 28 * 28, 5)  # 8, 8

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 56, 56))  # 16, 16
        # print(x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

CNNS = {
    "cifar10": CIFAR10_CNN,
    "chexpert": CheXpert_CNN,
    "eyepacs": EYEPACS_CNN,
    "eyepacs_tensors": EYEPACS_CNN,
    "chexpert_tensors": CheXpert_CNN,
}