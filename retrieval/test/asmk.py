import time
import torch

import  retrieval.utils.evaluation.asmk as eval_asmk
from retrieval.utils.general     import htime
from retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map


# logger
import logging
logger = logging.getLogger("retrieval")


def test_asmk(dataset, query_dl, db_dl, model, descriptor_size, ground_truth, asmk):
             
    # 
    if dataset in ["roxford5k", "rparis6k"]:
        revisited = True
    else:
        revisited = False

    #
    with torch.no_grad():
        
        # 
        start = time.time()
            
        # database indexing 
        logger.info('{%s}: extracting descriptors for database images', dataset)
        asmk_db = eval_asmk.index_database(db_dl, model, asmk)
            
        # query indexing
        logger.info('{%s}: extracting descriptors for query images', dataset)
        ranks = eval_asmk.query_ivf(query_dl, model, asmk_db)
            
        # scores
        if revisited:
            score = compute_map_revisited(ranks, ground_truth)
        else:
            score = compute_map(ranks, ground_truth)
        
        # time 
        logger.info('{%s}: running time = %s', dataset, htime(time.time() - start))

        return score