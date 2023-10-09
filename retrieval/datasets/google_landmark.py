import pickle
from os import path

from loguru import logger

from .tuples_dataset import TuplesDataset


class GoogleLandmarkDataset(TuplesDataset):
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

        # root
        data_path = path.join(data_path, name)
        ims_root = path.join(data_path, 'train')
        db_fn = path.join(data_path, f'{name}.pkl')

        if not path.exists(db_fn):
            return [], {}

        with open(db_fn, 'rb') as f:
            db = pickle.load(f)[mode]

        # setting fullpath for images
        self.images = [path.join(ims_root, db['cids'][i]+'.jpg')
                       for i in range(len(db['cids']))]

        # indices
        self.clusters = db['cluster']
        self.query_pool = db['qidxs']
        self.positive_pool = db['pidxs']

        #
        self.query_size = min(self.query_size, len(self.query_pool))
        self.pool_size = min(self.pool_size, len(self.images))
