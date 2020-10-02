import torch.nn as nn
from .modelutils import conv_layer, ResBlock, DenseBlock, AdaptivePooling, Lambda, flatten


class ModelScratch(nn.Module):
    """
    Model Scratch is a custom architecture for dog vs human classification
    along with dog breed classification. The architecture has 135
    activations as outputs; 133 for dog breeds and 2 for human vs dog each.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                      conv_layer(3, 16, ks=3, s=2, p=1),  # 112
                      DenseBlock(16, 8),
                      conv_layer(24, 64, ks=3, s=2, p=1),  # 56
                      DenseBlock(64, 64),
                      conv_layer(128, 256, ks=3, s=2, p=1),  # 28
                      ResBlock(256),
                      conv_layer(256, 512, ks=3, s=2, p=1),  # 14
                      AdaptivePooling(1),
                      Lambda(flatten),
                      nn.Linear(512, 135),
                      nn.BatchNorm1d(135)
        )

    def forward(self, x):
        return self.layers(x)
