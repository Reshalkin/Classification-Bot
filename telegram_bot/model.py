!pip install -q Pillow==4.1.1
!pip install -q "fastai==0.7.0"
!pip install -q torchtext==0.2.3
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python

from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
!apt update && apt install -y libsm6 libxext6

import os
accelerator = 'cu80' if os.path.exists('/opt/bin/nvidia-smi') else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
!pip install -q image
import pathlib
import gc
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import numpy as np
from PIL import Image
import shutil
import pickle
from skimage import io
from tqdm import tqdm, tqdm_notebook
from pathlib import Path
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from imageio import imread
from skimage.transform import resize
from skimage import color

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
!cp ../flowers/resnet34/resnet34.pth /root/.torch/models/resnet34-333f7ec4.pth
#cache_dir, models_dir


class ClassPredictor:
    def __init__(self):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.PATH = '../flowers/'
		data = ImageClassifierData.from_paths(self.PATH, tfms=tfms_from_model(resnet34, 224))
		self.model = ConvLearner.pretrained(resnet34, data, precompute=False)
		self.model.load('weights_bot')


    def predict(self, img_stream):
		_, val_tfms = tfms_from_model(resnet34,100)
		img = val_tfms(Image.open(img_stream))
		return np.argmax(self.model.predict_array(img[None]))