# import pickle
# import numpy as np
# import os
# from   tqdm import tqdm
# import yaml
# import json

# import torch
# from torch.utils.data import DataLoader

# from asmk import asmk_method, io_helpers, ASMKMethod, kernel as kern_pkg


# from retrieval.datasets import ImagesFromList, ImagesTransform, INPUTS

# from retrieval.utils.evaluation.ParisOxfordEval import compute_map


# # from loguru import logger
# 

# PARAM_PATH="./retrieval/configuration/defaults/asmk.yml"




# def compute_map_and_log(dataset, ranks, gnd, kappas=(1, 5, 10), log_debug=None):
#     """
#         Computed mAP and log it
    
#     :param str dataset: Dataset to compute the mAP on (e.g. roxford5k)
#     :param np.ndarray ranks: 2D matrix of ints corresponding to previously computed ranks
#     :param dict gnd: Ground-truth dataset structure
#     :param list kappas: Compute mean precision at each kappa
#     :param logging.Logger logger: If not None, use it to log mAP and all mP@kappa
#     :return tuple: mAP and mP@kappa (medium difficulty for roxford5k and rparis6k)
#     """
#     # new evaluation protocol
#     if dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):

#         # Easy
#         gnd_t = []
#         for gndi in gnd:
#             g           = {}
#             g['ok']     = np.concatenate([gndi['easy']])
#             g['junk']   = np.concatenate([gndi['junk'], gndi['hard']])
#             gnd_t.append(g)
            
#         mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

#         # Medium
#         gnd_t = []
#         for gndi in gnd:
#             g = {}
#             g['ok'] = np.concatenate([gndi['easy'], gndi['hard']])
#             g['junk'] = np.concatenate([gndi['junk']])
#             gnd_t.append(g)
        
#         mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

#         # Hard
#         gnd_t = []
#         for gndi in gnd:
#             g = {}
#             g['ok'] = np.concatenate([gndi['hard']])
#             g['junk'] = np.concatenate([gndi['junk'], gndi['easy']])
#             gnd_t.append(g)
            
#         mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)
        
#         # logging
#         log_debug("{%s}: mAP E: {%f}, M: {%f}, H: {%f}",
#                  dataset,
#                  np.around(mapE*100, decimals=2),
#                  np.around(mapM*100, decimals=2),
#                  np.around(mapH*100, decimals=2))

#         log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
#                  dataset,
#                  kappas[0],
#                  np.around(mprE * 100, decimals=2)[0],
#                  np.around(mprM * 100, decimals=2)[0],
#                  np.around(mprH * 100, decimals=2)[0])

#         log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
#                  dataset,
#                  kappas[1],
#                  np.around(mprE * 100, decimals=2)[1],
#                  np.around(mprM * 100, decimals=2)[1],
#                  np.around(mprH * 100, decimals=2)[1])

#         log_debug("{%s}: mP@k{%f} E: {%f}, M: {%f}, H: {%f}",
#                  dataset,
#                  kappas[2],
#                  np.around(mprE * 100, decimals=2)[2],
#                  np.around(mprM * 100, decimals=2)[2],
#                  np.around(mprH * 100, decimals=2)[2])

#         scores = {
#             "map_easy":     mapE.item(),        "mp@k_easy":    mprE,
#             "map_medium":   mapM.item(),        "mp@k_medium":  mprM,
#             "map_hard":     mapH.item(),        "mp@k_hard":    mprH
#                   }
        
#         return scores
    
#     else:
#         map_score, ap_scores, prk, pr_scores = compute_map(ranks, gnd, kappas=kappas)

#         log_debug("{%s}: mAP{%f}, mP@k: {%f} {%f} {%f}",
#                  dataset,
#                  np.around(map_score * 100, decimals=2),
#                  np.around(prk * 100, decimals=2)[0],
#                  np.around(prk * 100, decimals=2)[1],
#                  np.around(prk * 100, decimals=2)[2])
                
#         scores = {"map": map_score, "mp@k": prk, "ap": ap_scores, "p@k": pr_scores}
        
#         return scores


