
from copy import deepcopy
from os import mkdir, path

import gdown
import numpy as np
import timm
import torch
import torch.nn as nn
from core.progress import tqdm_progress
from core.registry.factory import load_state_dict, load_pretrained
from core.registry.register import get_pretrained_cfg
from core.transforms import tfn_image_net
from loguru import logger

from retrieval.models.base import RetrievalBase
from retrieval.models.modules.pools import GeM
from retrieval.utils.pca import PCA
from retrieval.utils.snapshot import save_snapshot

from .misc import _cfg, register_retrieval


def set_batchnorm_eval(m):
    """Set batchnorm layers to eval mode"""
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class GemHead(nn.Module):
    """ Generalized Mean Pooling head """

    def __init__(self, features_dim, out_dim, p=3.0):
        super(GemHead, self).__init__()

        self._features_dim = features_dim
        self._out_dim = out_dim

        # pooling
        self.pool = GeM(p=p)

        # whitening
        self.whiten = nn.Linear(features_dim, out_dim, bias=True)

    def forward(self, x, do_whitening=True):

        # pooling
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # whithen
        if do_whitening:
            x = self.whiten(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        return {'features': x}


class GemNet(RetrievalBase):

    def __init__(self, cfg):
        super(GemNet, self).__init__(cfg=cfg)

        # out dim
        self._out_dim = self.cfg.out_dim

        # create backbone FIXME: add pretrained correctly
        self.body = timm.create_model(self.cfg.backbone,
                                      features_only=True,
                                      out_indices=self.cfg.feature_scales,
                                      pretrained=True)

        # features dim
        self._features_dim = self.body.feature_info.channels()[-1]

        # create head
        self.head = GemHead(features_dim=self._features_dim,
                            out_dim=self.cfg.out_dim,
                            p=self.cfg.p)

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters, set to eval"""

        self.training = mode
        for module in self.children():
            module.train(mode)

        # freeze batchnorm
        self.apply(set_batchnorm_eval)

    def parameter_groups(self):
        """Return torch parameter groups"""

        # base
        LR = self.cfg.optimizer.lr
        WEIGHT_DECAY = self.cfg.optimizer.weight_decay

        # base layer
        layers = [self.body, self.head.whiten]

        # base params
        params = [{
            'params':          [p for p in x.parameters() if p.requires_grad],
            'lr':              LR,
            'weight_decay':    WEIGHT_DECAY} for x in layers]

        # 10x faster, no regularization
        if self.head.pool:
            params.append({
                'params':       [p for p in self.head.pool.parameters() if p.requires_grad],
                'lr':           10*LR,
                'weight_decay': 0.}
            )

        return params

    @torch.no_grad()
    def init_model(self, sample_dl, save_path=None, **kwargs):
        """Initialize model"""

        logger.info(
            f'PCA {self._features_dim}--{self._out_dim} for {len(sample_dl)}')

        # eval
        if self.training:
            self.eval()

        # progress bar
        progress_bar = tqdm_progress(
            sample_dl, colour='white', desc='extract global'.rjust(15))

        # extract vectors
        vecs = []
        for _, data in enumerate(progress_bar):

            # upload data
            data = {"image": data["image"].cuda()}

            # extract
            pred = self.forward(data, do_whitening=False)

            # append
            vecs.append(pred['features'].cpu().numpy())

            del pred

        # stack
        vecs = np.vstack(vecs)

        logger.info('compute PCA, this may take a while')

        m, P = PCA(vecs)
        m, P = m.T, P.T

        # create layer
        layer = deepcopy(self.head.whiten)
        data_size = layer.weight.data.size()
        num_d = layer.weight.shape[0]

        # project and shift
        projection = torch.Tensor(P[: num_d, :]).view(data_size)
        projected_shift = - \
            torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()

        # set layer
        layer.weight.data = projection
        layer.bias.data = projected_shift[:num_d]

        # save layer if needed
        if save_path is not None:
            save_path = path.join(save_path, "whiten.pth")
            torch.save(layer.state_dict(), save_path)
            logger.info(f"save whiten layer: {save_path}")

        # load layer if exsis already
        logger.info("load whiten layer")
        self.head.whiten.load_state_dict(layer.state_dict())
        logger.info("pca done")

    def transform_inputs(self, data: dict) -> dict:
        """Transform inputs"""

        # add data dim
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # normalize image net
        data["image"] = tfn_image_net(data["image"])

        return data

    def forward(self, data, do_whitening=True):
        """Forward pass"""

        # transform inputs
        data = self.transform_inputs(data)

        # body
        x = self.body(data["image"])
        x = x[-1]

        # head
        return self.head(x, do_whitening)

    @torch.no_grad()
    def extract(self, image):
        """Extract features from an image"""
        return self.forward(image, do_whitening=True)


default_cfgs = {
    'sfm_resnet50_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1gFRNJPILkInkuCZiCHqjQH_Xa2CUiAb5',
             backbone="resnet50", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
    'sfm_resnet50_c4_gem_1024':
        _cfg(drive='https://drive.google.com/uc?id=1ber3PbTF4ZWAmnBuJu5AEp2myVJFNM7F',
             backbone="resnet50", feature_scales=[1, 2, 3], out_dim=1024, p=3.0),
    'sfm_resnet101_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=10CqmzZE_XwRCyoiYlZh03tfYk1jzeziz',
             backbone="resnet101", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
    'sfm_resnet101_c4_gem_1024':
        _cfg(drive='https://drive.google.com/uc?id=1uYYuLqqE9TNgtmQtY7Mg2YEIF9VkqAYz',
             backbone="resnet101", feature_scales=[1, 2, 3], out_dim=1024, p=3.0),
    'gl18_resnet50_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1AaS4aXe2FYyi-iiLetF4JAo0iRqKHQ2Z',
             backbone="resnet50", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
}


def load_from_drive(variant, pretrained_drive):
    # name hub folder
    save_folder = "hub"
    save_path = save_folder + "/" + variant + ".pth"

    # create fodler
    if not path.exists(save_folder):
        mkdir(save_folder)

    # download from gdrive if weights not found
    if not path.exists(save_path):
        save_path = gdown.download(
            pretrained_drive, save_path, quiet=False, use_cookies=False)

    #  load from drive
    state_dict = load_state_dict(save_path)

    return state_dict


def _create_model(variant, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(variant)
    cfg = {**model_cfg, **cfg}

    # create model
    model = GemNet(cfg=cfg)

    # load pretrained weights
    if pretrained:

        # load_pretrained(model, variant, cfg)
        state_dict = load_from_drive(variant, cfg["drive"])["state_dict"]
        model.body.load_state_dict(state_dict["body"])
        model.head.load_state_dict(state_dict["head"])
    
    # TODO: we need to delete this, just to fix save_snapshot bug
    
    # load snapshot
    snapshot_path = f"hub/{variant}.pth"

    # check if snapshot exists
    if not path.exists(snapshot_path):
        logger.info(f"snapshot not found: {snapshot_path}")
        exit(0)
    snapshot = torch.load(snapshot_path, map_location="cpu")

    # save snapshot
    snapshot_path_new = f"hub/{variant}_new.pth"
    print(snapshot_path_new)
    config = snapshot["config"]

    # from training_meta
    epoch = snapshot["training_meta"]["epoch"]
    last_score = snapshot["training_meta"]["last_score"]
    best_score = snapshot["training_meta"]["best_score"]
    global_step = snapshot["training_meta"]["global_step"]

    # from state_dict
    optimizer = snapshot["state_dict"]["optimizer"]
    loss = snapshot["state_dict"]["loss"]
    total_loss = snapshot["state_dict"]["total_loss"]
    data_time = snapshot["state_dict"]["data_time"]
    batch_time = snapshot["state_dict"]["batch_time"]

    #
    save_snapshot(snapshot_path_new, config, epoch,
                  last_score, best_score, global_step,
                  model=model.state_dict(),
                  optimizer=optimizer,
                  loss=loss,
                  total_loss=total_loss,
                  data_time=data_time,
                  batch_time=batch_time)
    exit(0)
    return model


@register_retrieval
def sfm_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_gem_2048', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet50_c4_gem_1024(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model, only 4 features scales
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_c4_gem_1024', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_gem_2048', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet101_c4_gem_1024(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model, only 4 features scales
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_c4_gem_1024', cfg, pretrained, **model_args)


@register_retrieval
def gl18_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a gl18 ResNet-50 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('gl18_resnet50_gem_2048', cfg, pretrained, **model_args)
