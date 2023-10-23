
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg
from core.transforms import tfn_image_net
from sklearn.neighbors import NearestNeighbors

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval


def make_encoder():
    encoder = models.vgg16(weights='IMAGENET1K_V1')
    layers = list(encoder.features.children())[:-2]

    # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
    for layer in layers[:-5]:
        for p in layer.parameters():
            p.requires_grad = False
    encoder = nn.Sequential(*layers)

    return encoder


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()

        #
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input

        # cluster
        self.conv = nn.Conv2d(
            dim, num_clusters, kernel_size=(1, 1), bias=vladv2)

        # centroids
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 is False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(
                    x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return {'features': vlad}


class NetVLADNet(RetrievalBase):
    # paper reference
    paper_ref = ["https://arxiv.org/abs/1511.07247"]

    # code reference
    code_ref = ["https://github.com/Nanne/pytorch-NetVlad"]

    def __init__(self, cfg):
        super(NetVLADNet, self).__init__(cfg=cfg)

        # encoder
        self.encoder = make_encoder()

        # pool
        self.pool = NetVLAD(num_clusters=self.cfg.num_clusters,
                            dim=self.cfg.enc_dim)

    def transform_inputs(self, data: dict) -> dict:
        """transform inputs"""

        # add data dim
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # normalize image net
        data["image"] = tfn_image_net(data["image"])

        return data

    def forward(self, data=None, **kwargs):
        """ forward pass"""

        # transform inputs
        data = self.transform_inputs(data)

        # encoder
        x = self.encoder(data["image"])

        # pool
        x = self.pool(x)

        return x

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data, do_whitening=True)


default_cfgs = {
    'vgg16_netvlad':
        _cfg(drive='https://drive.google.com/uc?id=1_D9RN8JYWyd17giEY0iNKgdhzBiN746C',
             backbone="vgg16", enc_dim=512, num_clusters=64),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = NetVLADNet(cfg=cfg)

    # load pretrained weights
    if pretrained:
        load_pretrained(model, name, cfg, state_key="state_dict")

    return model


@register_retrieval
def vgg16_netvlad(cfg=None, pretrained=True, **kwargs):
    """Constructs a NetVLAD model with Pitts250k weights"""
    return _create_model('vgg16_netvlad', cfg, pretrained, **kwargs)
