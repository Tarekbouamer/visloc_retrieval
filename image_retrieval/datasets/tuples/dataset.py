import os
from  torch.utils.data import Dataset

import numpy as np
from PIL import Image

INPUTS = ["img"]

class ImagesFromList(Dataset):
    """ImagesFromList
        generic dataset from list of images
    """

    def __init__(self, root, images, bbxs=None, transform=None):

        images_fn = [os.path.join(root, images[i]) for i in range(len(images))]

        if len(images_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))

        self.root = root
        self.images = images
        
        self.images_fn = images_fn
        
        self.bbxs = bbxs
        self.transform = transform

    def __len__(self):
        return len(self.images_fn)

    def load_img(self, img_path):
      
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
          
        return img
      
    def __getitem__(self, item):

        img_path = self.images_fn[item]
        
        img = self.load_img(img_path)
        
        # crop if box exsists 
        if self.bbxs is not None:
            img = img.crop(self.bbxs[item])

        # perform transformation
        if self.transform is not None:
            out = self.transform(img)

        return out