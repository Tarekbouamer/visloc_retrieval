import time

import numpy as np
import torch
from core.device import to_cuda, to_numpy
from core.progress import tqdm_progress
from loguru import logger

from .base import BaseExtractor


class GlobalExtractor(BaseExtractor):
    """ Global feature extractor """

    def __init__(self, cfg, model_name=None, model=None):
        super(GlobalExtractor, self).__init__(cfg, model_name, model)

    @torch.no_grad()
    def extract(self, dataset, save_path=None, **kwargs):

        # features
        features = []

        # dataloader
        dataloader = self.make_dataloader(dataset)

        # time
        start_time = time.time()

        # progress bar
        progress_bar = tqdm_progress(
            dataloader, colour='magenta', desc='extract global'.rjust(15))

        # extract
        for _, data in enumerate(progress_bar):

            # to cuda
            to_cuda(data)

            # extract
            desc = self.model.extract(data)
            desc = desc['features'][0]

            # numpy
            desc = to_numpy(desc)

            # append
            features.append(desc)

        # stack features
        features = np.vstack(features)

        logger.success(
            f'extraction done {(time.time() - start_time):.4} seconds saved to {save_path}')

        return features

    def __repr__(self) -> str:
        msg = f"GlobalExtractor: {self.model.__class__.__name__}"
        return msg
