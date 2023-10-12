from copy import deepcopy
from os import path
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional
from core.progress import tqdm_progress
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg
from loguru import logger

from retrieval.models.base import RetrievalBase
from retrieval.utils.pca import PCA

from .misc import _cfg, register_retrieval


def l2n(x, eps=1e-6):
    """L2-normalize columns of x"""
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


class L2Attention(nn.Module):
    """ L2 attention """

    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)


class SmoothingAvgPooling(nn.Module):
    """ Smoothing average pooling"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        pad = self.kernel_size // 2
        return functional.avg_pool2d(x, self.kernel_size, stride=1, padding=pad,
                                     count_include_pad=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'

# FIXME: this is not used


class ConvDimReduction(nn.Conv2d):
    """ Convolutional dimension reduction"""

    def __init__(self, inp_dim, out_dim):
        super().__init__(inp_dim, out_dim, (1, 1), padding=0, bias=True)


class HowHead(nn.Module):
    """ How head"""

    def __init__(self, inp_dim, out_dim=128, kernel_size=3):
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # attention
        self.attention = L2Attention()

        # pool
        self.pool = SmoothingAvgPooling(kernel_size=kernel_size)

        # whiten
        self.whiten = nn.Conv2d(
            inp_dim, out_dim, kernel_size=(1, 1), padding=0, bias=True)

        # reset parameters
        self.reset_parameters()

        # not trainable
        for param in self.whiten.parameters():
            param.requires_grad = False

    def reset_parameters(self):

        for name, mod in self.named_modules():

            if isinstance(mod,  nn.Linear):
                nn.init.xavier_normal_(mod.weight, 1.0)

            elif isinstance(mod, nn.LayerNorm):
                nn.init.constant_(mod.weight, 0.01)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x, do_whitening=True):
        """ forward pass"""

        # attention
        attn = self.attention(x)

        # pool and reduction
        x = self.pool(x)

        # whiten
        if do_whitening:
            x = self.whiten(x)

        return {'features': x, 'attns': attn}


class HowNet(RetrievalBase):

    def __init__(self, cfg):
        super(HowNet, self).__init__(cfg=cfg)

        self._out_dim = self.cfg.out_dim

        # backbone
        self.body = timm.create_model(self.cfg.backbone,
                                      features_only=True,
                                      out_indices=self.cfg.feature_scales,
                                      pretrained=True)

        # features dim
        self._features_dim = self.body.feature_info.channels()[-1]

        # create head
        self.head = HowHead(inp_dim=self._features_dim,
                            out_dim=self._out_dim,
                            kernel_size=self.cfg.kernel_size)

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

            # upload batch
            pred = self.extract_locals(
                data, do_whitening=False, num_features=400)

            # append
            vecs.append(pred['features'].cpu().numpy())
            del pred

        # stack
        vecs = np.vstack(vecs)

        logger.info('Compute PCA, this may take a while')

        m, P = PCA(vecs)
        m, P = m.T, P.T

        # create layer
        layer = deepcopy(self.head.whiten)
        layer.weight.data.size()
        num_d = layer.weight.shape[0]

        # project and shift
        projection = torch.Tensor(P[: num_d, :]).unsqueeze(-1).unsqueeze(-1)
        projected_shift = - \
            torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()

        # set layer
        layer.weight.data = projection
        layer.bias.data = projected_shift[:num_d]

        # save layer if needed
        if save_path is not None:
            layer_path = path.join(save_path, "whiten.pth")
            torch.save(layer.state_dict(), layer_path)
            logger.info(f"Save whiten layer: {save_path}")

        # load layer if exsis already
        logger.info("Load whiten layer")
        self.head.whiten.load_state_dict(layer.state_dict())

        # not trainable
        logger.info("Freeze whiten layer")
        for param in self.head.whiten.parameters():
            param.requires_grad = False

        logger.success("PCA done")

    def parameter_groups(self, cfg):
        """
            Return torch parameter groups
        """

        # base
        LR = cfg.optimizer.lr
        WEIGHT_DECAY = cfg.optimizer.weight_decay

        # base layer
        layers = [self.body, self.head.attention, self.head.pool]

        # base params
        params = [{
            'params':          [p for p in x.parameters() if p.requires_grad],
            'lr':              LR,
            'weight_decay':    WEIGHT_DECAY} for x in layers]

        # freeze whiten
        if self.head.whiten:
            params.append({
                'params':       [p for p in self.head.whiten.parameters() if p.requires_grad],
                'lr':           0.0
            })

        return params

    def how_select_local(self, ms_features, ms_masks, scales, num_features):
        """ Convert multi-scale feature maps with attentions to a list of local descriptors
            Args:
                ms_features: list of feature maps at different scales
                ms_masks: list of attention maps at different scales
                scales: list of scales
                num_features: number of local descriptors to keep
            Returns:
                dict: dictionary with the following keys:
                    - features: local descriptors
                    - attns: attention weights
                    - locs: locations of local descriptors
                    - cls: scales of local descriptors
        """
        device = ms_features[0].device

        size = sum(x.shape[0] * x.shape[1] for x in ms_masks)

        desc = torch.zeros(
            size, ms_features[0].shape[1], dtype=torch.float32, device=device)
        atts = torch.zeros(size, dtype=torch.float32, device=device)
        locs = torch.zeros(size, 2, dtype=torch.int16, device=device)
        scls = torch.zeros(size, dtype=torch.float16, device=device)

        pointer = 0

        for sc, vs, ms in zip(scales, ms_features, ms_masks):

            #
            if len(ms.shape) == 0:
                continue

            #
            height, width = ms.shape
            numel = torch.numel(ms)
            slc = slice(pointer, pointer+numel)
            pointer += numel

            #
            desc[slc] = vs.squeeze(0).reshape(vs.shape[1], -1).T
            atts[slc] = ms.reshape(-1)
            width_arr = torch.arange(width, dtype=torch.int16)
            locs[slc, 0] = width_arr.repeat(height).to(device)  # x axis
            height_arr = torch.arange(height, dtype=torch.int16)
            # y axis
            locs[slc, 1] = height_arr.view(-1, 1).repeat(1,
                                                         width).reshape(-1).to(device)
            scls[slc] = sc

        #
        keep_n = min(
            num_features, atts.shape[0]) if num_features is not None else atts.shape[0]
        idx = atts.sort(descending=True)[1][:keep_n]

        #
        preds = {
            'features':    desc[idx],
            'attns':    atts[idx],
            'locs':     locs[idx],
            'cls':      scls[idx]
        }

        #
        return preds

    def weighted_spoc(self, ms_features, ms_weights):
        """Weighted SPoC pooling, summed over scales
        Args:
            ms_features: list of feature maps at different scales
            ms_weights: list of weights at different scales
        Returns:
            dict: dictionary with the following keys:
                - features: weighted SPoC pooling of the input feature maps
        """

        desc = torch.zeros(
            (1, ms_features[0].shape[1]), dtype=torch.float32, device=ms_features[0].device)

        for features, weights in zip(ms_features, ms_weights):
            desc += (features * weights).sum((-2, -1)).squeeze()

        desc = l2n(desc)

        preds = {
            'features': desc
        }

        return preds

    def _forward(self, data, scales=[1], do_whitening=True):
        features_list, attns_list = [], []

        # extract features at different scales
        for s in scales:
            imgs = functional.interpolate(
                data["image"], scale_factor=s, mode='bilinear', align_corners=False)

            # body
            x = self.body(imgs)
            if isinstance(x, List):
                x = x[-1]

            # head
            preds = self.head(x, do_whitening=do_whitening)

            #
            features_list.append(preds['features'])
            attns_list.append(preds['attns'])

        # normalize to max weight
        mx = max(x.max() for x in attns_list)
        attns_list = [x/mx for x in attns_list]

        return features_list, attns_list

    def extract_locals(self, image, scales=[1], num_features=1000, do_whitening=True):
        """Extract local descriptors from an image"""
        feat_list, attns_list = self._forward(image, scales, do_whitening)
        return self.how_select_local(feat_list, attns_list, scales, num_features)

    def forward(self, data, do_whitening=True):
        """Forward pass"""
        feat_list, attns_list = self._forward(data, do_whitening=do_whitening)
        return self.weighted_spoc(feat_list, attns_list)

    @torch.no_grad()
    def extract(self, data, scales=[1]):
        """Extract features from an image
            1. extract features & attentions
            3. weighted spoc
        """
        feat_list, attns_list = self._forward(data, scales, True)
        return self.weighted_spoc(feat_list, attns_list)


default_cfgs = {
    'sfm_resnet18_how_128':
        _cfg(drive='https://drive.google.com/uc?id=10o3xfP3piVoW3XDeSZLW6lErANRDZ61P',
             backbone="resnet18", feature_scales=[1, 2, 3, 4], out_dim=128, kernel_size=3),
    'sfm_resnet50_c4_how_128':
        _cfg(drive='https://drive.google.com/uc?id=1wy3tMPOq-tSYBnUssWACiPNRTVjSbdzb',
             backbone="resnet50", feature_scales=[1, 2, 3], out_dim=128, kernel_size=3)
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = HowNet(cfg=cfg)

    # load pretrained weights
    if pretrained:
        load_pretrained(model, name, cfg, state_key="model")

    return model


@register_retrieval
def sfm_resnet18_how_128(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model"""
    return _create_model('sfm_resnet18_how_128', cfg, pretrained, **kwargs)


@register_retrieval
def sfm_resnet50_c4_how_128(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with How model, only 4 features scales"""
    return _create_model('sfm_resnet50_c4_how_128', cfg, pretrained, **kwargs)
