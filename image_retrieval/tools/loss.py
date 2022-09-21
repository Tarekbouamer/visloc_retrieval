from asyncio.log import logger
from os import makedirs, path
from tqdm import tqdm
import shutil

import numpy as np

import  torch 
import  torch.nn as nn 
from    torch.utils.data import DataLoader, SubsetRandomSampler

# core
import core.backbones   as models
from core.backbones.url     import model_urls, model_urls_cvut
from core.backbones.util    import load_state_dict_from_url, init_weights
from core.utils.misc        import norm_act_from_config, freeze_params


# image retrieval
from image_retrieval.datasets.generic import ImagesFromList, ImagesTransform, INPUTS
from image_retrieval.modules.heads.head         import RetrievalHead

import image_retrieval.modules.necks   as necks

from image_retrieval.algos.algo                 import globalLoss
from image_retrieval.models.GF_net              import ImageRetrievalNet

from image_retrieval.modules.losses import TripletLoss, ContrastiveLoss, APLoss

# logger
import logging
logger = logging.getLogger("retrieval")

def build_loss(cfg):
    
    # parse params with default values
    global_config = cfg["global"]

    # Create Loss
    logger.debug("creating Loss function { %s }", global_config.get("loss"))
    
    loss_name = global_config.get("loss")
    loss_margin  =  global_config.getfloat("loss_margin")
    
    # Triplet
    if loss_name == 'triplet':
        return TripletLoss(margin=loss_margin)
    # Constractive   
    elif loss_name == 'contrastive':
        return ContrastiveLoss(margin=loss_margin)
    else:
        raise NotImplementedError(f"loss not implemented yet {global_config.get('loss') }" )