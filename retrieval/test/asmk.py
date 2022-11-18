import time
import torch

import  retrieval.utils.evaluation.asmk as eval_asmk
from retrieval.utils.general     import htime
from retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map
import numpy as np


# logger
import logging
logger = logging.getLogger("retrieval")

def test_asmk(dataset, query_dl, db_dl, feature_extractor, descriptor_size, ground_truth, asmk):
             
    # 
    if dataset in ["roxford5k", "rparis6k"]:
        revisited = True
    else:
        revisited = False

    # 
    start = time.time()
            
    # database indexing 
    logger.info('{%s}: extracting descriptors for database images', dataset)
    asmk_db = eval_asmk.index_database(db_dl, feature_extractor, asmk)
            
    # query indexing
    logger.info('{%s}: extracting descriptors for query images', dataset)
    ranks = eval_asmk.query_ivf(query_dl, feature_extractor, asmk_db)
    
    # 
    assert ranks.shape[0] == len(db_dl) and ranks.shape[1] == len(query_dl)
            
    # scores
    if revisited:
        score = compute_map_revisited(ranks, ground_truth)
    else:
        score = compute_map(ranks, ground_truth)
        
    # time 
    logger.info('{%s}: running time = %s', dataset, htime(time.time() - start))

    return score