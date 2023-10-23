

import torch
import torch.nn as nn
import torchvision
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg
from core.transforms import tfn_image_net

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval
from .modules import L2N, MAC, GeM, GeMmp, SPoC

POOLING = {
    'mac': MAC,
    'spoc': SPoC,
    'gem': GeM,
    'gemmp': GeMmp,
    # 'rmac': RMAC,
}

OUTPUT_DIM = {
    'alexnet':  256,
    'vgg11':  512,
    'vgg13':  512,
    'vgg16':  512,
    'vgg19':  512,
    'resnet18':  512,
    'resnet34':  512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,  # largest densenet
    'squeezenet1_0':  512,
    'squeezenet1_1':  512,
}


def get_backbone(backbone: str) -> torch.nn.Module:
    """ get feature extractor """

    net_in = getattr(torchvision.models, backbone)(pretrained=True)

    if backbone.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif backbone.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif backbone.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif backbone.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif backbone.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError(
            'Unsupported or unknown architecture: {}!'.format(backbone))

    return nn.Sequential(*features)


class ImageRetrievalNet(RetrievalBase):
    """Image Retrieval Network"""
    # reference paper
    paper_ref = ["https://arxiv.org/abs/1711.02512",
                 "https://arxiv.org/abs/1604.02426"]

    # reference code
    code_ref = ["https://github.com/filipradenovic/cnnimageretrieval-pytorch"]

    def __init__(self, cfg,):
        super(ImageRetrievalNet, self).__init__(cfg=cfg)

        self.features = get_backbone(backbone=self.cfg.backbone)

        dim = OUTPUT_DIM[self.cfg.backbone]

        # pooling
        if self.cfg.pool == 'gemmp':
            self.pool = POOLING[self.cfg.pool](mp=dim)
        else:
            self.pool = POOLING[self.cfg.pool]()

        # whiten
        self.whiten = nn.Linear(dim, dim, bias=True)

        # normlization
        self.norm = L2N()

    def transform_inputs(self, data: dict) -> dict:
        """transform inputs"""

        # add data dim
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # normalize image net
        data["image"] = tfn_image_net(data["image"])

        return data

    def forward(self, data):
        """Forward pass"""

        # transform inputs
        data = self.transform_inputs(data)

        # x -> features
        x = self.features(data["image"])

        # features -> pool -> norm
        x = self.norm(self.pool(x)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            x = self.norm(self.whiten(x))

        return {'features': x}

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data)


default_cfgs = {
    'sfm_resnet50_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
             backbone="resnet50", pool="gem",  out_dim=2048),
    'sfm_resnet101_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
             backbone="resnet101", pool="gem",  out_dim=2048),
    'sfm_resnet152_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
             backbone="resnet152", pool="gem",  out_dim=2048),
    'gl18_resnet50_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
             backbone="resnet50", pool="gem",  out_dim=2048),
    'gl18_resnet101_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
             backbone="resnet101", pool="gem",  out_dim=2048),
    'gl18_resnet152_gem':
        _cfg(url='http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
             backbone="resnet152", pool="gem",  out_dim=2048),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = ImageRetrievalNet(cfg=cfg)

    # load pretrained weights
    if pretrained:
        load_pretrained(model, name, cfg, state_key="state_dict")

    return model


@register_retrieval
def sfm_resnet50_gem(cfg=None, pretrained=True, **kwargs):
    """ sfm_resnet50_gem """
    return _create_model('sfm_resnet50_gem', cfg, pretrained, **kwargs)


@register_retrieval
def sfm_resnet101_gem(cfg=None, pretrained=True, **kwargs):
    """ sfm_resnet101_gem """
    return _create_model('sfm_resnet101_gem', cfg, pretrained, **kwargs)


@register_retrieval
def sfm_resnet152_gem(cfg=None, pretrained=True, **kwargs):
    """ sfm_resnet152_gem """
    return _create_model('sfm_resnet152_gem', cfg, pretrained, **kwargs)


@register_retrieval
def gl18_resnet50_gem(cfg=None, pretrained=True, **kwargs):
    """ gl18_resnet50_gem """
    return _create_model('gl18_resnet50_gem', cfg, pretrained, **kwargs)


@register_retrieval
def gl18_resnet101_gem(cfg=None, pretrained=True, **kwargs):
    """ gl18_resnet101_gem """
    return _create_model('gl18_resnet101_gem', cfg, pretrained, **kwargs)


@register_retrieval
def gl18_resnet152_gem(cfg=None, pretrained=True, **kwargs):
    """ gl18_resnet152_gem """
    return _create_model('gl18_resnet152_gem', cfg, pretrained, **kwargs)
