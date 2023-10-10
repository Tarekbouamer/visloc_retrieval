import time

import numpy as np
import torch
from core.device import to_cuda, to_numpy
from core.progress import tqdm_progress
from loguru import logger

from retrieval.extractors.base import BaseExtractor


class LocalExtractor(BaseExtractor):
    """ Local feature extractor """

    def __init__(self, cfg, model_name=None, model=None):
        super().__init__(cfg, model_name, model)

    @torch.no_grad()
    def extract(self, dataset, num_features=1000, scales=[1.0], save_path=None, **kwargs):

        # features
        features, imids = [], []

        # dataloader
        dataloader = self.make_dataloader(dataset)

        # time
        start_time = time.time()

        # progress bar
        progress_bar = tqdm_progress(
            dataloader, colour='green', desc='extract locals'.rjust(15))

        # extract
        for it, data in enumerate(progress_bar):

            # to cuda
            to_cuda(data)

            # prepare inputs
            image = self.transform_inputs(data["image"], **kwargs)

            # extract locals
            preds = self.model.extract_locals(image, num_features=num_features)
            desc = preds['features']

            # numpy
            desc = to_numpy(desc)

            # append
            features.append(desc)
            imids.append(np.full((desc.shape[0],), it))

        # stack
        features = np.vstack(features)
        ids = np.hstack(imids)

        #
        logger.info(
            f'Extraction done {(time.time() - start_time):.4} seconds saved to {save_path}')

        return features, ids

    def __repr__(self) -> str:
        msg = f"LocalExtractor: {self.model.__class__.__name__}"
        return msg
