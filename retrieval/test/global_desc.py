
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


def test_global_descriptor(dataset, query_dl, db_dl, feature_extractor, descriptor_size, ground_truth, scales=[1]):
    
    # extract query
    q_out   = feature_extractor.extract_global(query_dl.dataset, save_path=None)
    q_vecs  = q_out['features']
    print(q_vecs.shape)
    # extract database
    db_out  = feature_extractor.extract_global(db_dl.dataset, save_path=None)
    db_vecs = db_out['features']
    print(db_vecs.shape)
    print(q_vecs)

    # options  
    # batch_size = 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 
    if dataset in ["roxford5k", "rparis6k"]:
        revisited = True
    else:
        revisited = False

    #
    # with torch.no_grad():
            
    #     # extract query vectors
    #     q_vecs = torch.zeros(len(query_dl), descriptor_size).cuda()

    #     for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):
            
    #         batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
            
    #         desc    = extract_ms(batch["img"], feature_extractor, descriptor_size, scales=scales)
                            
    #         q_vecs[it * batch_size: (it+1) * batch_size, :] = desc
            
            # del desc

        # # extract database vectors
        # db_vecs = torch.zeros(len(db_dl), descriptor_size).cuda()

        # for it, batch in tqdm(enumerate(db_dl), total=len(db_dl)):

        #     batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}

        #     desc    = extract_ms(batch["img"], feature_extractor, descriptor_size, scales=scales)
                
        #     db_vecs[it * batch_size: (it+1) * batch_size, :] = desc

        #     del desc
                
    # convert to numpy
    # q_vecs  = q_vecs.cpu().numpy()
    # db_vecs = db_vecs.cpu().numpy()
            
    # search, rank, and print
    scores = np.dot(db_vecs, q_vecs.T)
    ranks  = np.argsort(-scores, axis=0)

    # scores
    if revisited:
        score = compute_map_revisited(ranks, ground_truth)
    else:
        score = compute_map(ranks, ground_truth)
    
    return score