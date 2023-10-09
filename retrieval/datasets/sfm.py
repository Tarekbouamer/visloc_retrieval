import pickle
from os import path

from loguru import logger

from retrieval.datasets.misc import cid2filename

from .tuples_dataset import TuplesDataset


class SfMDataset(TuplesDataset):
    def __init__(self, data_path, name, cfg={}, mode="train", transform=None):
        super().__init__(data_path, name, cfg=cfg, mode=mode, transform=transform)

        # build dataset
        self.build_dataset(data_path, name, mode)

        logger.info(
            f'SfMDataset "{name}" on {mode} mode with {len(self)} tuples')

    def build_dataset(self, data_path, name, mode):
        """Builds the dataset"""

        # logger
        logger.info(f'Building {name} {mode} dataset from {data_path}')

        # setting up paths
        db_root = path.join(data_path, 'train', name)
        ims_root = path.join(db_root, 'ims')
        db_fn = path.join(db_root, f'{name}.pkl')

        with open(db_fn, 'rb') as f:
            db = pickle.load(f)[mode]

        # get images full path
        self.images = [cid2filename(db['cids'][i], ims_root)
                       for i in range(len(db['cids']))]

        # indices
        self.clusters = db['cluster']
        self.query_pool = db['qidxs']
        self.positive_pool = db['pidxs']

        #
        self.query_size = min(self.query_size, len(self.query_pool))
        self.pool_size = min(self.pool_size, len(self.images))
