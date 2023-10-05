import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))

    def __repr__(self):
        return f'{self.__class__.__name__}'


class SPoC(nn.Module):
    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    def __repr__(self):
        return f'{self.__class__.__name__}'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.item():.4})'


class GeMmp(nn.Module):
    def __init__(self, p=3, mp=1, eps=1e-6):
        super(GeMmp, self).__init__()
        self.mp = mp
        self.p = Parameter(torch.ones(self.mp) * p).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.item()}, mp={self.mp.item()})'
