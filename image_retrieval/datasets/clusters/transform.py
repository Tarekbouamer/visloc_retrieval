import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tfn


class ISSTransform:

    def __init__(self,
                 shortest_size=None,
                 longest_max_size=None,
                 rgb_mean=None,
                 rgb_std=None,
                 random_flip=False,
                 random_scale=None):

        self.shortest_size = shortest_size

        self.longest_max_size = longest_max_size
        
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        
        self.random_flip = random_flip
        self.random_scale = random_scale

    def _adjusted_scale(self, in_width, in_height, target_size):
        max_size = max(in_width, in_height)
        
        scale = target_size / max_size

        return scale

    @staticmethod
    def _random_flip(img, ):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return img
        else:
            return img

    def _random_target_size(self):
        if len(self.random_scale) == 2:
            target_size = random.uniform(self.shortest_size * self.random_scale[0],
                                         self.shortest_size * self.random_scale[1])
        else:
            target_sizes = [self.shortest_size * scale for scale in self.random_scale]
            target_size = random.choice(target_sizes)

        return int(target_size)

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img
    
    def __call__(self, img, bbx=None):

        # Crop  bbx
        if bbx is not None:
            img = img.crop(box=bbx)

        # Random flip
        if self.random_flip:
            img = self._random_flip(img)

        scale = self._adjusted_scale(img.size[0], img.size[1], self.longest_max_size)

        out_size = tuple(int(dim * scale) for dim in img.size)
                
        img = img.resize(out_size, resample=Image.BILINEAR)

        # Image transformations
        img = tfn.to_tensor(img)

        # Normalize
        img = self._normalize_image(img)

        
        return dict(img=img)


class ISSTestTransform:
    def __init__(self,
                 shortest_size=None,
                 longest_max_size=None,
                 rgb_mean=None,
                 rgb_std=None,
                 random_scale=None):

        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size

        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self.random_scale = random_scale

    def _adjusted_scale(self, in_width, in_height, target_size):

        max_size = max(in_width, in_height)
        
        scale = target_size / max_size

        return scale

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img
    
    def __call__(self, img, bbx=None):

        # Crop  bbx
        if bbx is not None:
            
            img = img.crop(box=bbx)

        if self.longest_max_size:

            scale = self._adjusted_scale(img.size[0], img.size[1], self.longest_max_size)

            out_size = tuple(int(dim * scale) for dim in img.size)

            img = img.resize(out_size, resample=Image.BILINEAR)

        # Image transformations
        img = tfn.to_tensor(img)

        # Normalize
        img = self._normalize_image(img)
        
        return dict(img=img)
