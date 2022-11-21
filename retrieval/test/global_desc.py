
from tqdm import tqdm
import numpy as np
import torch

from retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map

from retrieval.datasets import INPUTS

import torch.nn.functional as functional

# logger
import logging
logger = logging.getLogger("retrieval")


def __check_size__(x, min_size=200, max_size=1200):
    # too large (area)
    if not (x.size(-1) * x.size(-2) <= max_size * max_size):
        return True
    # too small
    if not (x.size(-1) >= min_size and x.size(-2) >= min_size):
        return True
    return False


def __to_numpy__(x):
    if len(x.shape) > 1:
        x = x.squeeze(0)
    if x.is_cuda:
        x = x.cpu()    
    return x.numpy() 
 
 
def extract_ms(img, model, out_dim, scales=[1], min_size=100, max_size=2000):
    
    #
    desc = torch.zeros(out_dim).to(img.device)
    num_scales = 0. 
    
    #
    for scale in scales:

        # scale
        if scale == 1.0:
            img_s = img
        else:
            img_s = functional.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)   

            # assert size within boundaries
            if __check_size__(img_s, min_size, max_size):
                continue
        
        #       
        num_scales += 1.0
                 
        # extract
        preds  = model.extract_global(img_s, do_whitening=True)
        desc_s = preds['feats'].squeeze(0)
                
        # accum
        desc += desc_s
    #
    desc = (1.0/num_scales) * desc
    desc = functional.normalize(desc, dim=-1)
    
    return desc


def test_global_descriptor(dataset, query_dl, db_dl, feature_extractor, descriptor_size, ground_truth, scales=[1.0]):
    
    # 
    if dataset in ["roxford5k", "rparis6k"]:
        revisited = True
    else:
        revisited = False
    
    # extract query
    q_out   = feature_extractor.extract_global(query_dl, save_path=None, scales=scales)

    # extract database
    db_out  = feature_extractor.extract_global(db_dl, save_path=None, scales=scales)
    
    #
    q_vecs  = q_out['features']
    db_vecs = db_out['features']
      
    # search, rank, and print
    scores = np.dot(db_vecs, q_vecs.T)
    ranks  = np.argsort(-scores, axis=0)

    # scores
    if revisited:
        score = compute_map_revisited(ranks, ground_truth)
    else:
        score = compute_map(ranks, ground_truth)
    
    return score