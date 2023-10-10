import pickle
import time
from os import path, remove

from core.meter import htime
from loguru import logger

import retrieval.test.asmk as eval_asmk
from retrieval.test.ap import compute_map, compute_map_revisited

try:
    from asmk import asmk_method, io_helpers
except ImportError:
    logger.warning("asmk not exported from thirdparty, try \
                    export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)")


PARAM_PATH = "./retrieval/configuration/defaults/asmk.yml"


def asmk_init(params_path=None):

    #
    params_path = PARAM_PATH

    # params
    params = io_helpers.load_params(params_path)

    # init asmk_method
    asmk = asmk_method.ASMKMethod.initialize_untrained(params)

    return asmk, params


def train_codebook(cfg, sample_dl, extractor, asmk, scales=[1.0], save_path=None):
    """
        train_codebook
    """

    # remove old book
    if path.exists(save_path):
        remove(save_path)
    #
    train_out = extractor.extract_locals(
        sample_dl, scales=scales, save_path=None)
    train_vecs = train_out["features"]

    # run training
    asmk = asmk.train_codebook(train_vecs, cache_path=save_path)

    3
    train_time = asmk.metadata['train_codebook']['train_time']
    logger.debug(f"codebook trained in {train_time:.2f}s")

    return asmk


def index_database(db_dl, feature_extractor, asmk, scales=[1.0], distractors_path=None):
    """ 
        Asmk aggregate database and build ivf
    """

    db_out = feature_extractor.extract_locals(db_dl, scales=scales)

    # stack
    db_vecs = db_out["features"]
    db_ids = db_out["ids"]

    # build ivf
    asmk_db = asmk.build_ivf(
        db_vecs, db_ids, distractors_path=distractors_path)

    index_time = asmk_db.metadata['build_ivf']['index_time']
    logger.debug(f"database indexing in {index_time:.2f}s")

    return asmk_db


def query_ivf(query_dl, feature_extractor, asmk_db, scales=[1.0], cache_path=None, imid_offset=0):
    """ 
        asmk aggregate query and build ivf
    """

    q_out = feature_extractor.extract_locals(query_dl, scales=scales)

    # stack
    q_vecs = q_out["features"]
    q_ids = q_out["ids"] + imid_offset

    # run ivf
    metadata, query_ids, ranks, scores = asmk_db.query_ivf(q_vecs, q_ids)
    logger.info(
        f"average query time (quant + aggr + search) is {metadata['query_avg_time']:.3f}s")

    #
    ranks = ranks.T

    if cache_path:
        with cache_path.open("wb") as handle:
            pickle.dump({"metadata": metadata, "query_ids": query_ids,
                        "ranks": ranks, "scores": scores}, handle)

    return ranks


def test_asmk(dataset, query_dl, db_dl, feature_extractor, scales, ground_truth=None, asmk=None):
    """
        Aggregated Selective Match Kernel test  
    """

    #
    revisited = True if dataset in ["roxford5k", "rparis6k"] else False

    #
    start = time.time()

    # database indexing
    logger.info('{%s}: extracting descriptors for database images', dataset)
    asmk_db = eval_asmk.index_database(
        db_dl, feature_extractor, asmk, scales=scales)

    # query indexing
    logger.info('{%s}: extracting descriptors for query images', dataset)
    ranks = eval_asmk.query_ivf(
        query_dl, feature_extractor, asmk_db, scales=scales)

    # assert
    assert ranks.shape[0] == len(db_dl) and ranks.shape[1] == len(query_dl)

    # scores
    if revisited:
        score = compute_map_revisited(ranks, ground_truth)
    else:
        score = compute_map(ranks, ground_truth)

    # time
    logger.info('{%s}: running time = %s', dataset, htime(time.time() - start))

    return score
