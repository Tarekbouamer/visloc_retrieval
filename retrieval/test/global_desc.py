
from tqdm import tqdm
import numpy as np
import torch

from retrieval.test.mean_ap import compute_map_revisited, compute_map

from retrieval.datasets import INPUTS

import torch.nn.functional as functional

# logger
from loguru import logger



def test_global_descriptor(dataset, query_dl, db_dl, extractor, ground_truth, scales=[1.0]):
    """ test global mode 
    
        dataset:                str                         name of test dataset
        query_dl:               data.Dataloader             query dataloader
        db_dl:                  data.Dataloader             database dataloader
        extractor:              FeatureExtractor            feature extractor
        ground_truth:           List                        ground truth 
        scales:                 List                        extraction scales
    """
    
    # 
    revisited = True if dataset in ["roxford5k", "rparis6k"] else False
    
    # extract query
    q_out   = extractor.extract_global(query_dl, save_path=None, scales=scales)

    # extract database
    db_out  = extractor.extract_global(db_dl, save_path=None, scales=scales)
    
    #
    q_vecs  = q_out['features']
    db_vecs = db_out['features']
      
    # rank
    scores = np.dot(db_vecs, q_vecs.T)
    ranks  = np.argsort(-scores, axis=0)

    # scores
    if revisited:
        scores = compute_map_revisited(ranks, ground_truth)
    else:
        scores = compute_map(ranks, ground_truth)
    
    return scores