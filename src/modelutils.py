import torch
import torch.nn as nn


def conv_layer(f_in, f_out, ks, s, p):
    return nn.Sequential(
        nn.Conv2d(f_in, f_out, kernel_size=ks, stride=s, padding=p,bias=False),
        nn.BatchNorm2d(f_out),
        nn.ReLU())


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        self.conv1 = conv_layer(nf, nf, 3, 1, 1)
        self.conv2 = conv_layer(nf, nf, 3, 1, 1)

    def forward(self, X):
        return X + self.conv2(self.conv1(X))


class DenseBlock(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.ni, self.nf = ni, nf
        self.conv1 = conv_layer(ni, nf, 3, 1, 1)
        self.conv2 = conv_layer(nf, nf, 3, 1, 1)

    def forward(self, X):
        return torch.cat([X, self.conv2(self.conv1(X))], dim=1)


class AdaptivePooling(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(ni)
        self.avg_pool = nn.AdaptiveAvgPool2d(ni)

    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        return max_pool + avg_pool


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def flatten(x):
    return x.view(x.shape[0], -1)
