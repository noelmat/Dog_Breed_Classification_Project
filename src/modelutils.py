import torch
import torch.nn as nn


def conv_layer(f_in, f_out, ks, s, p):
    """
    Returns a Conv2D followed by a BatchNorm2d and ReLU for the input
    params.

    Args:
        f_in: number of input channels
        f_out: number of output channels
        ks: kernel size
        s: stride
        p: padding size
    """
    return nn.Sequential(
        nn.Conv2d(f_in, f_out, kernel_size=ks, stride=s, padding=p,
                  bias=False),
        nn.BatchNorm2d(f_out),
        nn.ReLU())


class ResBlock(nn.Module):
    """
    ResBlock for a given number of filters. It atwo conv_layers.
    Args:
        nf: number of filters.
    """
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        self.conv1 = conv_layer(nf, nf, 3, 1, 1)
        self.conv2 = conv_layer(nf, nf, 3, 1, 1)

    def forward(self, X):
        return X + self.conv2(self.conv1(X))


class DenseBlock(nn.Module):
    """
    DenseBlock for a given number of filters. It concatenates conv_layer
    outputs with the input features.
    Args:
        ni: number of input filters.
        nf: number of output filters.
    """
    def __init__(self, ni, nf):
        super().__init__()
        self.ni, self.nf = ni, nf
        self.conv1 = conv_layer(ni, nf, 3, 1, 1)
        self.conv2 = conv_layer(nf, nf, 3, 1, 1)

    def forward(self, X):
        return torch.cat([X, self.conv2(self.conv1(X))], dim=1)


class AdaptivePooling(nn.Module):
    """
    Adaptive Pooling adds the activations from a AdaptiveMaxPool2d and
    AdaptiveAvgPool2d.
    Args:
        ni: number of features.
    """
    def __init__(self, ni):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(ni)
        self.avg_pool = nn.AdaptiveAvgPool2d(ni)

    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        return max_pool + avg_pool


class Lambda(nn.Module):
    """
    Custom Layer that takes in a function and creates a nn.Module
    Args:
        func: function to be applied during forward pass.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def flatten(x):
    """
    Returns a flattened version of the input.
    """
    return x.view(x.shape[0], -1)
