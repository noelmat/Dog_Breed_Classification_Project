import mimetypes
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models

from matplotlib import pyplot as plt
Path.ls = lambda x: list(x.iterdir())
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate