import numpy as np
import time
import shutil
import torch

import os

from timm import utils

# 
from image_retrieval.tools.model             import build_model, run_pca
from image_retrieval.tools.optimizer         import build_optimizer, build_lr_scheduler
from image_retrieval.tools.dataloader        import build_train_dataloader, build_val_dataloader, build_sample_dataloader
from image_retrieval.tools.loss              import build_loss

from image_retrieval.tools.events            import EventWriter
from image_retrieval.tools.evaluation        import DatasetEvaluator


from image_retrieval.utils.logging                 import  _log_api_usage
from image_retrieval.utils.configurations          import config_to_string
from image_retrieval.utils.snapshot     import save_snapshot, resume_from_snapshot


# logger
import logging
logger = logging.getLogger("retrieval")



class FeatureExtractorOptions(object):
    pass


class FeatureExtractor():
    def __init__(self, args, cfg, eval=False, dataset=None):
        super().__init__()
        
        # options
        self.options = FeatureExtractorOptions()
        
        if dataset is None:
            raise ValueError(f'dataset is None Type {dataset}')
          
        # dataset
        self.dataset = dataset
          
        # cfg 
        logger.info("\n %s", config_to_string(cfg))
        self.cfg          = cfg
        self.args         = args
        
        # build  
        self.model  = build_model(cfg)
        
        # load
        self.load_snapshot()
                    
    
    def load_snapshot(self):
        """
        """
        
        # model
        snapshot_path = os.path.join(self.options.path)
        logger.info(f"load model {snapshot}")

        # load
        snapshot = resume_from_snapshot(self.model, snapshot_path, ["body", "head"])
        del snapshot_last  
              
    def extract(self):
      
        for data in self.dataset:
            print(data)
        
  