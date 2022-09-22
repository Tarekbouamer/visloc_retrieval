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
                 preprocessing=None,
                 augmentation=None,
                 postprocessing=None,
                 mean=None,
                 std=None,
                 is_aug=False
                 ):
        
        self.max_size = max_size
        self.mean   = mean
        self.std    = std

       
        # transformations 
        self.preprocessing  = preprocessing
        self.augmentation   = augmentation
        self.postprocessing = postprocessing
        
        if postprocessing is None:
            self.postprocessing = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])
        
        self.is_aug = is_aug

    def __call__(self, img):

        
        if self.is_aug:
            img = self.preprocessing(img)
            img = self.augmentation(img)

        else:
            # testing 
            img.thumbnail((self.max_size, self.max_size), Image.BILINEAR)        
        
        img = self.postprocessing(img)           

        return dict(img=img)

