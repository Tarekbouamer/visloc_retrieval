import time

import numpy as np
import torch
from core.device import get_device, to_cuda, to_numpy
from core.progress import tqdm_progress
from loguru import logger
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from retrieval.models import create_retrieval


class LocalExtractor():
    """ Local feature extractor """

    # device
    device = get_device()

    def __init__(self, model_name=None, model=None, cfg=None):
        super().__init__()
        #
        self.cfg = cfg

        # model
        if model is None:
            assert model_name is not None, "model name or model must be provided"
            model = create_retrieval(model_name, pretrained=True)

        # eval mode
        self.model = model.eval().to(self.device)

        # TODO: remove this and add it to base model
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
            img = self.transform_inputs(data['img'], **kwargs)

            # extract locals
            preds = self.model.extract_locals(img, num_features=num_features)
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
            f'extraction done {(time.time() - start_time):.4} seconds saved to {save_path}')

        return features, ids

    def __repr__(self) -> str:
        msg = f"LocalExtractor: {self.model.__class__.__name__}"
        return msg
