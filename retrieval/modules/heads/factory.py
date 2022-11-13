import torch
from .registry import is_model, model_entrypoint

from torch.hub import load_state_dict_from_url
import gdown

import os

# logger
import logging
logger = logging.getLogger("retrieval")


def create_head(
        model_name,
        inp_dim, 
        out_dim,
        **kwargs):
    """Create a head 
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
   
    model = create_fn(inp_dim, out_dim,**kwargs)

    return model
    