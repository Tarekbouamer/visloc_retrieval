import time
import torch

import  retrieval.utils.evaluation.asmk as eval_asmk
from retrieval.utils.general     import htime
from retrieval.utils.evaluation.ParisOxfordEval import compute_map_revisited, compute_map
import numpy as np


# logger
import logging
logger = logging.getLogger("retrieval")

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)  ]
        print(pos)
        print(pos.shape)

        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj) ]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map_and_log(dataset, ranks, gnd, kappas=(1, 5, 10), logger=None):
    """Computed mAP and log it
    :param str dataset: Dataset to compute the mAP on (e.g. roxford5k)
    :param np.ndarray ranks: 2D matrix of ints corresponding to previously computed ranks
    :param dict gnd: Ground-truth dataset structure
    :param list kappas: Compute mean precision at each kappa
    :param logging.Logger logger: If not None, use it to log mAP and all mP@kappa
    :return tuple: mAP and mP@kappa (medium difficulty for roxford5k and rparis6k)
    """
    # new evaluation protocol
    if dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        gnd_t = []
        for gndi in gnd:
            g = {}
            g['ok'] = np.concatenate([gndi['easy']])
            g['junk'] = np.concatenate([gndi['junk'], gndi['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for gndi in gnd:
            g = {}
            g['ok'] = np.concatenate([gndi['easy'], gndi['hard']])
            g['junk'] = np.concatenate([gndi['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for gndi in gnd:
            g = {}
            g['ok'] = np.concatenate([gndi['hard']])
            g['junk'] = np.concatenate([gndi['junk'], gndi['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        if logger:
            fmap = lambda x: np.around(x*100, decimals=2)
            logger.info(f"Evaluated {dataset}: mAP E: {fmap(mapE)}, M: {fmap(mapM)}, H: {fmap(mapH)}")
            logger.info(f"Evaluated {dataset}: mP@k{kappas} E: {fmap(mprE)}, M: {fmap(mprM)}, H: {fmap(mprH)}")

        scores = {"map_easy": mapE.item(), "mp@k_easy": mprE, "ap_easy": apsE, "p@k_easy": prsE,
                  "map_medium": mapM.item(), "mp@k_medium": mprM, "ap_medium": apsM, "p@k_medium": prsM,
                  "map_hard": mapH.item(), "mp@k_hard": mprH, "ap_hard": apsH, "p@k_hard": prsH}
        return scores

    # old evaluation protocol
    map_score, ap_scores, prk, pr_scores = compute_map(ranks, gnd, kappas=kappas)
    if logger:
        fmap = lambda x: np.around(x*100, decimals=2)
        logger.info(f"Evaluated {dataset}: mAP {fmap(map_score)}, mP@k {fmap(prk)}")
    return {"map": map_score, "mp@k": prk, "ap": ap_scores, "p@k": pr_scores}


def test_asmk(dataset, query_dl, db_dl, feature_extractor, descriptor_size, ground_truth, asmk):
             
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
        asmk_db = eval_asmk.index_database(db_dl, feature_extractor, asmk)
            
        # query indexing
        logger.info('{%s}: extracting descriptors for query images', dataset)
        ranks = eval_asmk.query_ivf(query_dl, feature_extractor, asmk_db)
            
        # scores
        # if revisited:
        #     score = compute_map_revisited(ranks, ground_truth)
        # else:
        score = compute_map_revisited(ranks.T, ground_truth)
        
        # time 
        logger.info('{%s}: running time = %s', dataset, htime(time.time() - start))

        return score