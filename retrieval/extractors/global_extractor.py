import time

import numpy as np
import torch
from core.device import get_device, to_cuda, to_numpy
from core.progress import tqdm_progress
from loguru import logger
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .base import BaseExtractor


class GlobalExtractor(BaseExtractor):
    """ Global feature extractor """

    # device
    device = get_device()

    def __init__(self, cfg, model_name=None, model=None):
        super().__init__(cfg, model_name, model)
        #
        self.cfg = cfg

        # transform
        self.transform = transforms.Compose([
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                 std=IMAGENET_DEFAULT_STD),
        ])

    def transform_inputs(self, x, **kwargs):

        #
        normalize = kwargs.pop('normalize', False)

        # normalize
        if normalize:
            x = self.transform(x)

        # BCHW
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x

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

            # prepare inputs
            img = self.transform_inputs(data['img'], **kwargs)

            # extract
            desc = self.model.extract(img)
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
