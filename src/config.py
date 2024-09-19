import os, sys
import time, math
import argparse, random
from math import exp
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as tfs
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as FF
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.models import vgg16

from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

steps = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resume = False
eval_step = 5000
learning_rate = 0.0001
pretrained_model_dir = '../input/ffa-net-for-single-image-dehazing-pytorch/trained_models/'
model_dir = './trained_models/'
trainset = 'its_train'
testset = 'its_test'
network = 'ffa'
gps = 3
blocks = 5
bs = 1
crop = True
crop_size = 240
no_lr_sche = True
perloss = True

model_name = f"{trainset}_{network.split('.')[0]}_{gps}_{blocks}"
pretrained_model_dir += model_name + '.pk'
model_dir += model_name + '.pk'
log_dir = f'logs/{model_name}'

for directory in ['trained_models', 'numpy_files', 'logs', 'samples']:
    if not os.path.exists(directory):
        os.mkdir(directory)

if not os.path.exists(f"samples/{model_name}"):
    os.mkdir(f'samples/{model_name}')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

crop_size = 'whole_img' if crop else crop_size