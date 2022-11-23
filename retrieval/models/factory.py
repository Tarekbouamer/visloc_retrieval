

import torch

import gdown
import os

from .registry import is_model, model_entrypoint
from torch.hub import load_state_dict_from_url

# logger
import logging
logger = logging.getLogger("retrieval")

# inspired timm.models

def load_state_dict(checkpoint_path, use_ema=True):
    """ load weights """
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        
        state_dict = checkpoint[state_dict_key]
        logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, variant, pretrained_cfg, strict=True):
    """ Load pretrained checkpoint either from local file, url or google drive
    
        model:              nn.Module  
        variant:            str             model name   
        pretrained_cfg:     dict            default configuration 
        strict:             boolean         
    """
    
    #
    pretrained_file   = pretrained_cfg.get('file',  None)
    pretrained_url    = pretrained_cfg.get('url',   None)
    pretrained_drive  = pretrained_cfg.get('drive', None)

    if pretrained_file:
        logger.info(f'Loading pretrained weights from file ({pretrained_file})')
        
        # load 
        state_dict = load_state_dict(pretrained_file)
    
    elif pretrained_url:
        logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        # load 
        state_dict = load_state_dict_from_url(pretrained_url, map_location='cpu', progress=True, check_hash=False)  
    
    elif pretrained_drive:
        #
        logger.info(f'Loading pretrained weights from google drive ({pretrained_drive})')        
        
        #
        save_folder = "pretrained_drive"
        save_path   = save_folder + "/" + variant + ".pth"

        # create fodler
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        # download from gdrive if weights not found    
        if not os.path.exists(save_path):
            save_path = gdown.download(pretrained_drive, save_path, quiet=False, use_cookies=False)
  
        #  load from drive
        state_dict = load_state_dict(save_path)
    else:
        logger.warning("No pretrained weights exist or were found for this model. Using random initialization.")
        return
    
    # load body and head weights
    model.body.load_state_dict(state_dict["body"], strict=strict)
    model.head.load_state_dict(state_dict["head"], strict=strict)
    
    
def create_model(model_name, cfg=None, pretrained=False, pretrained_cfg=None, **kwargs):
    """ create a model
    
        model_name:         str             model name
        cfg:                dict            config 
        pretrained:         boolean         load model weights 
        pretrained_cfg:     dict            default config 
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # check if model exsists
    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    #
    create_fn = model_entrypoint(model_name)
    
    # create model
    model = create_fn(cfg=cfg, pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)

    return model