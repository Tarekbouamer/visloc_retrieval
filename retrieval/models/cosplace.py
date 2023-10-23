from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg
from core.transforms import tfn_image_net
from loguru import logger
from torch.nn.parameter import Parameter

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval

CHANNELS_NUM_IN_LAST_CONV = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "vgg16": 512,
}


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


def get_pretrained_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[
                                 f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(
            weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(
            pretrained=True)
    return model


def get_backbone(backbone_name: str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("resnet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logger.debug(
            f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        # Remove avg pooling and FC layer
        layers = list(backbone.children())[:-2]

    elif backbone_name == "vgg16":
        layers = list(backbone.features.children())[
            :-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.debug(
            "Train last layers of the VGG-16, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim


class CosPlace(RetrievalBase):
    """CosPlace model"""
    # paper reference
    paper_ref = ["https://arxiv.org/abs/2204.02287",
                 "https://arxiv.org/abs/2308.10832"]
    # code reference
    code_ref = ["https://github.com/gmberton/CosPlace",
                "https://github.com/gmberton/EigenPlaces"]

    def __init__(self, cfg):
        super(CosPlace, self).__init__(cfg=cfg)
        assert self.cfg.backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(self.cfg.backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, self.cfg.out_dim),
            L2Norm()
        )

    def transform_inputs(self, data: dict) -> dict:
        """transform inputs"""

        # add data dim
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # normalize image net
        data["image"] = tfn_image_net(data["image"])

        return data

    def forward(self, data):

        #  transform inputs
        data = self.transform_inputs(data)

        # extract features
        x = self.backbone(data["image"])

        # aggregation
        x = self.aggregation(x)

        return {"features": x}

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data)


default_cfgs = {
    # cosplaces
    'cosplace_vgg16_gem_512':
        _cfg(drive='https://drive.google.com/uc?id=1F6CT-rnAGTTexdpLoQYncn-ooqzJe6wf',
             backbone="vgg16",  out_dim=512),
    'cosplace_resnet18_gem_512':
        _cfg(drive='https://drive.google.com/uc?id=1rQAC2ZddDjzwB2OVqAcNgCFEf3gLNa9U',
             backbone="resnet18",  out_dim=512),
    'cosplace_resnet50_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA',
             backbone="resnet50",  out_dim=2048),
    'cosplace_resnet101_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1PF7lsSw1sFMh-Bl_xwO74fM1InyYy1t8',
             backbone="resnet101",  out_dim=2048),
    'cosplace_resnet152_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1AlF7xPSswDLA1TdhZ9yTVBkfRnJm0Hn8',
             backbone="resnet152",  out_dim=2048),

    # eigenplaces
    'eigenplace_vgg16_gem_512':
        _cfg(url='https://github.com/gmberton/EigenPlaces/releases/download/v1.0/VGG16_512_eigenplaces.pth',
             backbone="vgg16",  out_dim=512),
    'eigenplace_resnet18_gem_2048':
        _cfg(url='https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet18_512_eigenplaces.pth',
             backbone="resnet18",  out_dim=512),
    'eigenplace_resnet50_gem_2048':
        _cfg(url='https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_2048_eigenplaces.pth',
             backbone="resnet50",  out_dim=2048),
    'eigenplace_resnet101_gem_2048':
        _cfg(url='https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet101_2048_eigenplaces.pth',
             backbone="resnet101",  out_dim=2048),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = CosPlace(cfg=cfg)

    # load pretrained weights
    if pretrained:
        load_pretrained(model, name, cfg)

    return model


@register_retrieval
def cosplace_vgg16_gem_512(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k VGG-16 with GeM model"""
    return _create_model('cosplace_vgg16_gem_512', cfg, pretrained, **kwargs)


@register_retrieval
def cosplace_resnet18_gem_512(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model"""
    return _create_model('cosplace_resnet18_gem_512', cfg, pretrained, **kwargs)


@register_retrieval
def cosplace_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model"""
    return _create_model('cosplace_resnet50_gem_2048', cfg, pretrained, **kwargs)


@register_retrieval
def cosplace_resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model"""
    return _create_model('cosplace_resnet101_gem_2048', cfg, pretrained, **kwargs)


@register_retrieval
def cosplace_resnet152_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-152 with GeM model"""
    return _create_model('cosplace_resnet152_gem_2048', cfg, pretrained, **kwargs)


@register_retrieval
def eigenplace_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    return _create_model('eigenplace_resnet50_gem_2048', cfg, pretrained, **kwargs)


@register_retrieval
def eigenplace_resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    return _create_model('eigenplace_resnet101_gem_2048', cfg, pretrained, **kwargs)
