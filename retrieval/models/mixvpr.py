
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg
from core.transforms import tfn_image_net
from torchvision import transforms

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=1024,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()
        self.in_h = in_h                    # height of input feature maps
        self.in_w = in_w                    # width of input feature maps
        self.in_channels = in_channels      # depth of input feature maps

        self.out_channels = out_channels    # depth wise projection dimension
        self.out_rows = out_rows            # row wise projection dimesion

        self.mix_depth = mix_depth          # L the number of stacked FeatureMixers
        # ratio of the mid projection layer in the mixer block
        self.mlp_ratio = mlp_ratio

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):

        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(
                    weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(
                    weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512

        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class VPRModel(RetrievalBase):
    # paper reference
    paper_ref = ["https://arxiv.org/abs/2303.02190"]

    # code reference
    code_ref = ["https://github.com/amaralibey/MixVPR"]

    def __init__(self, cfg):
        super(VPRModel, self).__init__(cfg=cfg)

        # backbone
        self.backbone = ResNet(model_name=self.cfg.backbone,
                               layers_to_crop=[4])

        # aggregator
        self.aggregator = MixVPR(in_channels=self.backbone.out_channels,
                                 out_channels=self.cfg.out_dim,
                                 out_rows=self.cfg.out_rows)

    def transform_inputs(self, data: dict) -> dict:
        """transform inputs"""

        # add data dim
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # resize image
        transform = transforms.Resize(
            size=(320, 320), interpolation=transforms.InterpolationMode.BILINEAR)

        data["image"] = transform(data["image"])

        # normalize image net
        data["image"] = tfn_image_net(data["image"])

        return data

    def forward(self, data):
        #  transform inputs
        data = self.transform_inputs(data)

        # extract features
        x = self.backbone(data["image"])

        # aggregate features
        x = self.aggregator(x)

        return {"features": x}

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data)


default_cfgs = {
    'mixvpr_resnet_128':
        _cfg(drive='https://drive.google.com/uc?id=1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz',
             backbone="resnet50",  out_dim=64, out_rows=2),
    'mixvpr_resnet_512':
        _cfg(drive='https://drive.google.com/uc?id=1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj',
             backbone="resnet50",  out_dim=256, out_rows=2),
    'mixvpr_resnet_4096':
        _cfg(drive='https://drive.google.com/uc?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L',
             backbone="resnet50",  out_dim=1024, out_rows=4),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = VPRModel(cfg=cfg)

    # load pretrained weights
    if pretrained:
        load_pretrained(model, name, cfg)

    return model


@register_retrieval
def mixvpr_resnet_128(cfg=None, pretrained=True, **kwargs):
    return _create_model('mixvpr_resnet_128', cfg, pretrained, **kwargs)


@register_retrieval
def mixvpr_resnet_512(cfg=None, pretrained=True, **kwargs):
    return _create_model('mixvpr_resnet_512', cfg, pretrained, **kwargs)


@register_retrieval
def mixvpr_resnet_4096(cfg=None, pretrained=True, **kwargs):
    return _create_model('mixvpr_resnet_4096', cfg, pretrained, **kwargs)
