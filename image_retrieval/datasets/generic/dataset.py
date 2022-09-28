import collections.abc as collections
from os import path

import torch
import torch.utils.data as data

import numpy as np

from pathlib import Path

from PIL import Image

from    torchvision.transforms import functional as tfn
import  torchvision.transforms as transforms

import cv2
import h5py

# logger
import logging
logger = logging.getLogger("retrieval")


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)

    return image


def list_h5_names(path):
    names = []
    
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
      
    return list(set(names))
  
class ImagesTransform:
    def __init__(self,
                 max_size,
                 mean=None,
                 std=None):
        
        self.max_size = max_size
        self.mean     = mean
        self.std      = std
       
        # self.tfn = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])

    def __call__(self, img):

        # resize
        if self.max_size:
            longest_size  = max(img.size[0], img.size[1])
            scale         = self.max_size / longest_size
            out_size      = tuple(int(dim * scale) for dim in img.size)
            img           = img.resize(out_size, resample=Image.BILINEAR)
        
        # to Tensor,         
        img = self.tfn(img)
        
        return dict(img=img)


class ImagesFromList(data.Dataset):
    
    def __init__(self, images_path, split="", max_size=None): 
        
        # 
        if max_size is None:
            raise ValueError(f'max_size is None Type {max_size}')

        # 
        self.max_size       = max_size
        self.split          = split        
        self.images_path    = images_path
        
        # load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.images_path).glob('**/'+ ext)) 
        
        #                                
        if len(paths) == 0:
            raise ValueError(f'Could not find any image in path: {self.images_path}.')
        
        #     
        self.images_fn = sorted(list(set(paths)))
        logger.info(f'found {len(self.images_fn)} images in {self.images_path}') 

    def __len__(self):
        return len(self.images_fn)

    def resize_image(self, img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)     
    
    def get_names(self):
        return [ self.split + "/" + str(p.relative_to(self.images_path)) for p in self.images_fn] 
         
    def load_img(self, img_path):        
        return cv2.imread(str(img_path), cv2.IMREAD_COLOR) 
    
    def __getitem__(self, item):
        #
        out = {}
        
        #
        img_path  = self.images_fn[item]
        
        # cv2
        img             = self.load_img(img_path)
        original_size   = np.array(img.shape[:2][::-1])

        # resize    
        if self.max_size :
            scale       = self.max_size / max(original_size)
            target_size = tuple(int(round(x * scale)) for x in original_size)
            img         = self.resize_image(img, target_size)

        # dict
        out["img"]              = img
        out["img_name"]         = str(self.split + "/" + str(img_path.relative_to(self.images_path)))
        out["original_size"]    = original_size
        
        return out