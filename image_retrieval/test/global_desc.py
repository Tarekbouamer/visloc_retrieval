
from tqdm import tqdm
import numpy as np
import torch

from image_retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map

from image_retrieval.datasets import INPUTS

# logger
import logging
logger = logging.getLogger("retrieval")


def test_global_descriptor(dataset, query_dl, db_dl, model, descriptor_size, ground_truth):
  
    # options  
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 
    if dataset in ["roxford5k", "rparis6k"]:
        revisited = True
    else:
        revisited = False

    #
    with torch.no_grad():
            
        # extract query vectors
        q_vecs = torch.zeros(len(query_dl), descriptor_size).cuda()

        for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):
            
            batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
            desc    = model(**batch, do_whitening=True)
                
            q_vecs[it * batch_size: (it+1) * batch_size, :] = desc
            del desc

        # extract database vectors
        db_vecs = torch.zeros(len(db_dl), descriptor_size).cuda()

        for it, batch in tqdm(enumerate(db_dl), total=len(db_dl)):

            batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
            desc    = model(**batch, do_whitening=True)
                
            # append
            db_vecs[it * batch_size: (it+1) * batch_size, :] = desc
            del desc
                
    # convert to numpy
    q_vecs  = q_vecs.cpu().numpy()
    db_vecs = db_vecs.cpu().numpy()
            
    # search, rank, and print
    scores = np.dot(db_vecs, q_vecs.T)
    ranks  = np.argsort(-scores, axis=0)

    # scores
    if revisited:
        score = compute_map_revisited(ranks, ground_truth)
    else:
        score = compute_map(ranks, ground_truth)
    
    return score