import time

from os import makedirs, path
from tqdm import tqdm

import numpy as np

import torch
import torch.utils.data as data

import  image_retrieval.utils.evaluation.asmk as eval_asmk
from image_retrieval.utils.general     import htime
from image_retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map

from image_retrieval.datasets import ImagesFromList, ImagesTransform, INPUTS, ParisOxfordTestDataset


TEST_MODES = ["global_descriptor", "ASMK", "All"]

# logger
import logging
logger = logging.getLogger("retrieval")


def build_paris_oxford_dataset(data_path, name_dataset, cfg):
    
    assert path.exists(data_path), logger.error("path: {data_path} does not exsists !!")
    
    logger.info(f'[{name_dataset}]: loading test dataset from {data_path}')
         
    db = ParisOxfordTestDataset(root_dir=data_path, name=name_dataset)
    
    # options 
    trans_opt = {   "max_size":     cfg["test"].getint("max_size")}
            
    dl_opt = {  "batch_size":   1, 
                "shuffle":      False, 
                "num_workers":  cfg["test"].getint("num_workers"), 
                "pin_memory":   True }  
    
    # query loader
    query_data  = ImagesFromList(root='', images=db['query_names'], bbxs=db['query_bbx'], transform=ImagesTransform(**trans_opt) )
    query_dl    = data.DataLoader( query_data, **dl_opt)
    
    # database loader
    db_data     = ImagesFromList( root='', images=db['img_names'], transform=ImagesTransform(**trans_opt) )  
    db_dl       = data.DataLoader(  db_data,  **dl_opt )
    
    # ground
    ground_truth = db['gnd']
    
    return query_dl, db_dl, ground_truth


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


def test(args, config, model, rank=None, world_size=None, logger=None, **varargs):
    
    # Eval mode
    model.eval()
    test_config = config["test"]
    
    mode = test_config.get("mode")
    assert mode in TEST_MODES, "Failed in test mode selection"

    logger.info('Evaluating network on test datasets { %s }', mode)

    # Evaluate on test datasets
    list_datasets = test_config.getstruct("datasets")
        
    if mode == "global_descriptor":
        scores, avg_score = test_global(args, config,  model, list_datasets, logger=logger, **varargs)
        
    elif mode == "ASMK":
        scores, avg_score = test_asmk(args, config, model, list_datasets, logger=logger, **varargs)
        
    elif mode == "All":
        scores, avg_score = test_global(args, config,  model, list_datasets, logger=logger, **varargs)
        scores, avg_score = test_asmk(args, config, model, list_datasets, logger=logger, **varargs)

          
    # As Evaluation metrics
    logger.info('Average score = %s', avg_score)

    return scores, avg_score
