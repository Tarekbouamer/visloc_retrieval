from os import makedirs, path
from tqdm import tqdm
import shutil
from copy import deepcopy

import numpy as np

import  torch 

# image retrieval
from retrieval.datasets import  INPUTS 

from retrieval.utils.io   import create_withen_file_from_cfg
from retrieval.utils.pca   import PCA

# logger
import logging
logger = logging.getLogger("retrieval")

def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
        

    
     
def compute_pca(args, cfg, model, sample_dl):
    
    # 
    body_cfg        = cfg["body"]
    global_cfg      = cfg["global"]
    data_cfg        = cfg["dataloader"]

    # Path    
    whithen_folder = path.join(args.directory, "whithen")

    if not path.exists(whithen_folder):
        logger.info("Save whithening layer: %s ", whithen_folder)
        makedirs(whithen_folder)
    
    # Whithen_path
    whithen_path = create_withen_file_from_cfg(cfg, whithen_folder, logger)

    # Avoid recomputing same layer for further experiments
    if ( not path.isfile(whithen_path) or global_cfg.getboolean("update") ):
                     
        # Compute layer
        whiten_layer = run_pca(model, sample_dl, device=varargs["device"], logger=logger)
        
        # Save layer to whithen_path
        logger.info("Save whiten layer: %s ", whithen_path)

        torch.save(whiten_layer.state_dict(), whithen_path)

    # load from whithen_path
    logger.info("Load whiten state: %s ", whithen_path)
    layer_state = torch.load(whithen_path, map_location="cpu")
    
    # Init model layer    
    model.init_whitening(layer_state, logger)
                