
import os

from torch.utils.data import DataLoader

from retrieval.datasets import PlaceDataset

FILE_NAMES = {
    "cph": {
        "query": "retrieval/configuration/msls/mapillarycph_imageNames_query.txt",
        "index": "retrieval/configuration/msls/mapillarycph_imageNames_index.txt",
    },
    "sf": {
        "query": "retrieval/configuration/msls/mapillarysf_imageNames_query.txt",
        "index": "retrieval/configuration/msls/mapillarysf_imageNames_index.txt",
    },
}


def build_msls_dataset(data_path, dataset, cfg):
    """ build Mapillary Street Level Sequences dataset """

    # assert
    assert dataset in FILE_NAMES.keys(), "dataset not found {}".format(dataset)

    # load file names
    q_name_path = FILE_NAMES[dataset]["query"]
    db_name_path = FILE_NAMES[dataset]["index"]

    # assert
    assert os.path.exists(q_name_path), "file not found {}".format(q_name_path)
    assert os.path.exists(db_name_path), "file not found {}".format(db_name_path)

    # build query dataset
    query_set = PlaceDataset(data_path, q_name_path, 
                             max_size=cfg.test.max_size
                             )

    query_dl = DataLoader(dataset=query_set,
                          num_workers=cfg.test.num_workers,
                          batch_size=cfg.test.batch_size,
                          shuffle=False)

    # build db dataset
    db_set = PlaceDataset(data_path, db_name_path, 
                          max_size=cfg.test.max_size
                          )

    db_dl = DataLoader(dataset=db_set,
                       num_workers=cfg.test.num_workers,
                       batch_size=cfg.test.batch_size,
                       shuffle=False)

    return query_dl, db_dl, None
