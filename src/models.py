from .modelutils import conv_layer, ResBlock, DenseBlock, \
                     AdaptivePooling, Lambda, flatten
from .imports import nn, models


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
                      nn.Dropout(p=0.05),
                      DenseBlock(16, 8),
                      nn.Dropout(p=0.05),
                      conv_layer(24, 64, ks=3, s=2, p=1),  # 56
                      nn.Dropout(p=0.05),            
                      DenseBlock(64, 64),
                      nn.Dropout(p=0.05),
                      conv_layer(128, 256, ks=3, s=2, p=1),  # 28
                      nn.Dropout(p=0.05),
                      ResBlock(256),
                      nn.Dropout(p=0.05),
                      conv_layer(256, 512, ks=3, s=2, p=1),  # 14
                      AdaptivePooling(1),
                      Lambda(flatten),
                      nn.Linear(512, 300),
                      nn.BatchNorm1d(300),
                      nn.ReLU(),
                      nn.Linear(300, 135),
                      nn.BatchNorm1d(135)
        )

    def forward(self, x):
        return self.layers(x)


class ModelTransfer(nn.Module):
    """
    Model Transfer is a resnet34 pretrained on Imagenet with a custom
    head for our specific data
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self.model.fc = nn.Linear(512, 300)
        self.head = nn.Sequential(nn.BatchNorm1d(300),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.05),
                                  nn.Linear(300, 135),
                                  nn.BatchNorm1d(135))

    def forward(self, x):
        x = self.head(self.model(x))
        return x
