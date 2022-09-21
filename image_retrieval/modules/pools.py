import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from image_retrieval.modules.normalizations import L2N


class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))
        # return F.adaptive_max_pool2d(x, (1,1)) # alternativ


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
    

class GeMmp(nn.Module):

    def __init__(self, p=3, mp=1, eps=1e-6):
        super(GeMmp, self).__init__()

        self.mp = mp
        self.p = Parameter(torch.ones(self.mp) * p).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps

    def forward(self, x):

        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):

        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)
        # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + self.eps).expand_as(v)

        for l in range(1, self.L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
                cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
            R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
            R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
            vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
            vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + self.eps).expand_as(vt)
            v += vt

        return v


class Rpool(nn.Module):
    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool, self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def roipool(self, x, rpool, L=3, eps=1e-6):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        _, idx = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        vecs = []
        vecs.append(rpool(x).unsqueeze(1))

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b).int() - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b).int() - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    vecs.append(rpool(x.narrow(2, i_, wl).narrow(3, j_, wl)).unsqueeze(1))

        return torch.cat(vecs, dim=1)

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = self.roipool(x, self.rpool, L=self.L, eps=self.eps)

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0]*s[1], s[2], s[3], s[4]) # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4]) # size: #im, #reg, D, 1, 1

        # aggregate regions into a single globalFeatures vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            o = self.norm(o.sum(1, keepdim=False)) # size: #im, D, 1, 1

        return o


POOLING_LAYERS = {
    "MAC": MAC,
    "SPoC": SPoC,
    "GeM": GeM,
    "GeMmp": GeMmp,
    "RMAC": RMAC,
    "ROIpool": Rpool
}


RET_LAYERS = [MAC, SPoC, GeM, GeMmp, RMAC, Rpool]
