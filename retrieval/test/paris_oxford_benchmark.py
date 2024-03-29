from os import path

from loguru import logger
from torch.utils.data import DataLoader

from retrieval.datasets import (
    ImagesFromList,
    ImagesTransform,
    ParisOxfordTestDataset,
)


def build_paris_oxford_dataset(data_path, name_dataset, cfg):
    """ Build paris oxford dataset """

    assert path.exists(data_path), \
        logger.error(f"{data_path} does not exsists !!")

    logger.info(f'[{name_dataset}]: loading test dataset from {data_path}')

    # dataset
    db = ParisOxfordTestDataset(root_dir=data_path, name=name_dataset)

    # options
    trans_opt = {"max_size":     cfg.test.max_size}

    # transform
    tfn = ImagesTransform(**trans_opt)

    # query loader
    query_data = ImagesFromList(root='', images=db['query_names'], bbxs=db['query_bbx'],
                                transform=tfn)
    query_dl = DataLoader(query_data, num_workers=4)

    # database loader
    db_data = ImagesFromList(root='', images=db['img_names'], 
                             transform=tfn)
    db_dl = DataLoader(db_data, num_workers=4)

    # ground
    ground_truth = db['gnd']

    return query_dl, db_dl, ground_truth
