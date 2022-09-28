from copy import deepcopy
import imghdr
from typing import List
import torch
import torch.nn as nn

from  tqdm import tqdm
 
import numpy as np
from collections import OrderedDict

from image_retrieval.datasets.tuples import ImagesFromList, ImagesTransform, INPUTS

    

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, body, head):
        super(ImageRetrievalNet, self).__init__()
        
        self.body   = body
        self.head   = head
        
    def forward(self, img=None, do_whitening=True):
          
        # body
        x = self.body(img)
        
        if isinstance(x, List):
            x = x[-1] 
        
        # head
        preds = self.head(x, do_whitening)
        
        if self.training:
            return preds

        return preds
        
