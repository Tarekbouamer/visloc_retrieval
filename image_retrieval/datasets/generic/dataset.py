
from torch.utils.data import Dataset

import numpy as np

from pathlib import Path
import cv2


# logger
import logging
logger = logging.getLogger("retrieval")


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

class ImagesListDataset(Dataset):
  
    def __init__(self, images_path, cameras_path=None, split='', max_size=1024): 
        
        self.images_path    = images_path
        self.cameras_path   = cameras_path
        self.split          = split  
        self.max_size       = max_size
           
        # Load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.images_path).glob('**/'+ ext)) 
        
        #                               
        if len(paths) == 0:
            raise ValueError(f'Could not find any image in path: {self.images_path}.')
        
        #   
        self.images_fn = sorted(list(set(paths)))      
        logger.info('Found {len(self.images_fn)} images in {self.images_path}') 
        
    def __len__(self):
        return len(self.images_fn)

    def resize_image(self, img, size):
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return  img     
    
    def get_names(self):
        return [ self.split + "/" + str(p.relative_to(self.images_path)) for p in self.images_fn] 
         
    def load_img(self, img_path, mode='color'):

        if    mode=='gray':
            mode = cv2.IMREAD_GRAYSCALE
        
        elif  mode=='color':
            mode = cv2.IMREAD_COLOR
        
        else:
            raise KeyError(f'mode {mode} not found!')
                
        return cv2.imread(str(img_path), mode) 
    
    def get_cameras(self,):
        if hasattr(self, "cameras"):
            return self.cameras     
        else:
            None          
    
    def __getitem__(self, item):
        #
        out = {}
        
        # load
        item_path  = self.images_fn[item]
        img = self.load_img(item_path)
        
        ori_size   = np.array(img.shape[:2][::-1])

        # resize    
        if self.max_size :
            scale = self.max_size / max(ori_size)
            target_size = tuple(int(round(x * scale)) for x in ori_size)
            img = self.resize_image(img, target_size)

        # dict
        out["img"]              = img
        out["img_name"]         = str(item_path.relative_to(self.images_path)).split('.')[0]
        out["original_size"]    = ori_size
        
        return out