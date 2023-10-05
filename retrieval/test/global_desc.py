
import numpy as np

from retrieval.test.mean_ap import compute_map, compute_map_revisited


def test_global(dataset, query_dl, db_dl, extractor, ground_truth, scales=[1.0]):
    """ test global descriptor """

    # revisited dataset
    revisited = True if dataset in ["roxford5k", "rparis6k"] else False

    # extract query
    q_vecs = extractor.extract(query_dl)

    # extract database
    db_vecs = extractor.extract(db_dl)

    # rank
    scores = np.dot(db_vecs, q_vecs.T)
    ranks = np.argsort(-scores, axis=0)

    # scores
    if revisited:
        scores = compute_map_revisited(ranks, ground_truth)
    else:
        scores = compute_map(ranks, ground_truth)

    return scores
