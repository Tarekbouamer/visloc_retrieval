from cmath import pi
import os
import torch.utils.data as data

import random

import numpy as np
from PIL import Image

from    torchvision.transforms import functional as tfn
import  torchvision.transforms as transforms


class ImagesTransform:

    def __init__(self,
                 max_size,
                 preprocessing =None,
                 augmentation=None,
                 postprocessing=None,
                 mean=None,
                 std=None,
                 is_train=False
                 ):
        
        self.max_size = max_size
        self.mean   = mean
        self.std    = std

        self.is_train = is_train
       
        # transformations 
        self.preprocessing  = preprocessing
        self.augmentation   = augmentation
        self.postprocessing = postprocessing
        
        if postprocessing is None:
            self.postprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img):

        
        if self.is_train:
            img = self.preprocessing(img)
            img = self.augmentation(img)

        else:
            img.thumbnail((self.max_size, self.max_size), Image.BILINEAR)        
        
        img = self.postprocessing(img)           

        return dict(img=img)

