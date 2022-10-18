import numpy as np
import time

import torch

import os

from image_retrieval.models.factory import create_model

from image_retrieval.utils.configurations          import config_to_string


# logger
import logging
logger = logging.getLogger("retrieval")



class FeatureExtractorOptions(object):
    pass


class FeatureExtractor():
    def __init__(self, model_name, args=None, cfg=None, eval=False, dataset=None):
        super().__init__()
        
        # options
        self.options = FeatureExtractorOptions()
        
        if dataset is None:
            raise ValueError(f'dataset is None Type {dataset}')
          
        # dataset
        self.dataset = dataset
          
        # cfg 
        self.cfg          = cfg
        self.args         = args
        
        # build  
        self.model  = create_model(model_name, pretrained=True)
        print(self.model)

              
    def extract(self):
      
        for data in self.dataset:
            print(data)
            
            
if __name__ == '__main__':
    
    create_fn = FeatureExtractor("resnet50_c4_gem_1024", dataset="test")
    print(create_fn)

        
  