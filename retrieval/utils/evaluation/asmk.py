import pickle
import numpy as np
import os
from   tqdm import tqdm
import yaml
import json

import torch
from torch.utils.data import DataLoader

from asmk import asmk_method, io_helpers, ASMKMethod, kernel as kern_pkg


from retrieval.datasets import ImagesFromList, ImagesTransform, INPUTS

from retrieval.utils.evaluation.ParisOxfordEval import compute_map


# logger
import logging
logger = logging.getLogger("retrieval")


PARAM_PATH="./retrieval/configuration/defaults/asmk.yml"


def asmk_init(params_path=None): 
    
    # load yml file
    if params_path is None:
        params_path = PARAM_PATH
    
    # params
    params = io_helpers.load_params(params_path)

    # init asmk_method
    asmk = asmk_method.ASMKMethod.initialize_untrained(params)
    
    #
    return asmk, params



def train_codebook(cfg, train_images, feature_extractor, asmk, save_path=None):
    """
        train_codebook
    """

    # if exsists, load
    if save_path and os.path.exists(save_path):
        return asmk.train_codebook(None, cache_path=save_path)
    
    # options 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    trans_opt = {   "max_size":     cfg["test"].getint("max_size")}
     
    dl_opt = { 
              "batch_size":   1,      
              "shuffle":      False, 
              "num_workers":  cfg["test"].getint("num_workers"), 
              "pin_memory":   True }  
    
    
    # train dataloader
    train_data  = ImagesFromList(root='', images=train_images, transform=ImagesTransform(**trans_opt) )                 
    train_dl    = DataLoader(train_data,  **dl_opt )
    
    train_out   = feature_extractor.extract_locals(train_dl, save_path=None)
    train_vecs  = train_out["features"]

    
    # with torch.no_grad():
         
    #     # extract vectors
    #     train_vecs = []

    #     for it, batch in tqdm(enumerate(train_dl), total=len(train_dl)):

    #         # batch
    #         batch = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
    #         print(batch)
    #         preds = feature_extractor.extract_locals(**batch, do_whitening=True)
            
    #         # append
    #         train_vecs.append(preds['feats'].cpu().numpy())   
            
    #         del preds
                
    #     # stack
    #     train_vecs  = np.vstack(train_vecs)

    # run 
    asmk = asmk.train_codebook(train_vecs, cache_path=save_path)
    train_time = asmk.metadata['train_codebook']['train_time']
    logger.debug(f"codebook trained in {train_time:.2f}s")
    
    return asmk
  

def index_database(db_dl, feature_extractor, asmk, distractors_path=None):
    """ 
            Asmk aggregate database and build ivf
    """
    
    db_out = feature_extractor.extract_locals(db_dl)
    
    # stack
    db_vecs  = db_out["features"]
    db_ids   = db_out["ids"]            

    # build ivf
    asmk_db = asmk.build_ivf(db_vecs, db_ids, distractors_path=distractors_path)
    
    index_time  = asmk_db.metadata['build_ivf']['index_time']
    ivf_stats   = asmk_db.metadata['build_ivf']['ivf_stats']
    
    logger.debug(f"database indexing in {index_time:.2f}s")
    
    for k, v in ivf_stats.items():
        logger.debug(f"ivf stats:   {k}:    {v:.2f}")
    
    return asmk_db
  
 
def query_ivf(query_dl, feature_extractor, asmk_db, cache_path=None, imid_offset=0):
    """ 
        asmk aggregate query and build ivf
    """

    q_out = feature_extractor.extract_locals(query_dl)

    # stack
    q_vecs  = q_out["features"]
    q_ids   = q_out["ids"] + imid_offset  
                 
                 
    # run ivf
    metadata, query_ids, ranks, scores = asmk_db.query_ivf(q_vecs, q_ids)
    logger.info(f"average query time (quant + aggr + search) is {metadata['query_avg_time']:.3f}s")
    
    # 
    ranks = ranks.T 
    
    if cache_path:
        with cache_path.open("wb") as handle:
            pickle.dump({"metadata": metadata, "query_ids": query_ids, "ranks": ranks, "scores": scores}, handle)
    
    return ranks


def compute_map_and_log(dataset, ranks, gnd, kappas=(1, 5, 10), log_debug=None):
    """
        Computed mAP and log it
    
    :param str dataset: Dataset to compute the mAP on (e.g. roxford5k)
    :param np.ndarray ranks: 2D matrix of ints corresponding to previously computed ranks
    :param dict gnd: Ground-truth dataset structure
    :param list kappas: Compute mean precision at each kappa
    :param logging.Logger logger: If not None, use it to log mAP and all mP@kappa
    :return tuple: mAP and mP@kappa (medium difficulty for roxford5k and rparis6k)
    """
    # new evaluation protocol
    if dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):

        # Easy
        gnd_t = []
        for gndi in gnd:
            g           = {}
            g['ok']     = np.concatenate([gndi['easy']])
            g['junk']   = np.concatenate([gndi['junk'], gndi['hard']])
            gnd_t.append(g)
            
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        # Medium
        gnd_t = []
        for gndi in gnd:
            g = {}
            g['ok'] = np.concatenate([gndi['easy'], gndi['hard']])
            g['junk'] = np.concatenate([gndi['junk']])
            gnd_t.append(g)
        
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        # Hard
        gnd_t = []
        for gndi in gnd:
            g = {}
            g['ok'] = np.concatenate([gndi['hard']])
            g['junk'] = np.concatenate([gndi['junk'], gndi['easy']])
            gnd_t.append(g)
            
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)
        
        # logging
        log_debug("{%s}: mAP E: {%f}, M: {%f}, H: {%f}",
                 dataset,
                 np.around(mapE*100, decimals=2),
                 np.around(mapM*100, decimals=2),
                 np.around(mapH*100, decimals=2))

        log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
                 dataset,
                 kappas[0],
                 np.around(mprE * 100, decimals=2)[0],
                 np.around(mprM * 100, decimals=2)[0],
                 np.around(mprH * 100, decimals=2)[0])

        log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
                 dataset,
                 kappas[1],
                 np.around(mprE * 100, decimals=2)[1],
                 np.around(mprM * 100, decimals=2)[1],
                 np.around(mprH * 100, decimals=2)[1])

        log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
                 dataset,
                 kappas[2],
                 np.around(mprE * 100, decimals=2)[2],
                 np.around(mprM * 100, decimals=2)[2],
                 np.around(mprH * 100, decimals=2)[2])

        scores = {
            "map_easy":     mapE.item(),        "mp@k_easy":    mprE,
            "map_medium":   mapM.item(),        "mp@k_medium":  mprM,
            "map_hard":     mapH.item(),        "mp@k_hard":    mprH
                  }
        
        return scores
    
    else:
        map_score, ap_scores, prk, pr_scores = compute_map(ranks, gnd, kappas=kappas)

        log_debug("{%s}: mAP{%f}, mP@k: {%f} {%f} {%f}",
                 dataset,
                 np.around(map_score * 100, decimals=2),
                 np.around(prk * 100, decimals=2)[0],
                 np.around(prk * 100, decimals=2)[1],
                 np.around(prk * 100, decimals=2)[2])
                
        scores = {"map": map_score, "mp@k": prk, "ap": ap_scores, "p@k": pr_scores}
        
        return scores


