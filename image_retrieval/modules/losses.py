from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Parameter

import math
import numpy as np

class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
            
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        # x is D x N
        x = x.permute(1, 0)
        dim = x.size(0) 
        
        # number of tuples
        nq = torch.sum(label.data==-1)
        
        # Number of images per tuple Nq+ Np+ Nn
        S = x.size(1) // nq 
        
        xp = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,    dim).permute(1,0)
        idx = [i for i in range(len(label)) if label.data[i] != -1]
        
        xn = x[:, idx]
        lbl = label[label!=-1]

        diff = xp - xn
        
        D = torch.pow(diff + self.eps, 2).sum(dim=0).sqrt()

        y = 0.5 * lbl * torch.pow(D, 2) + 0.5 * (1 - lbl) * torch.pow(torch.clamp(self.margin - D, min=0), 2)
        
        y = torch.sum(y)
        
        return y


class TripletLoss(nn.Module):
  
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        
        self.margin = margin

    def forward(self, x, target):

        # x is N x D
        dim = x.size(1)
                
        # Number of tuples            
        nq = torch.sum(target.data==-1).item() 
        
        # Number of images per tuple Nq+ Np+ Nn
        S = x.size(0) // nq 

        xa = x[target.data==-1, :].repeat(1,S-2).view((S-2)*nq,  dim)
        xp = x[target.data==1,  :].repeat(1,S-2).view((S-2)*nq,  dim)
        xn = x[target.data==0,  :]


        dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=1)
        dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=1)

        return torch.sum(torch.clamp(dist_pos - dist_neg + self.margin, min=0))

    
class APLoss (nn.Module):
    """ 
        Differentiable AP loss, through quantization. From the paper: https://arxiv.org/abs/1906.07589
        
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        
        self.nq = nq
        
        self.min = min
        self.max = max
        
        gap = max - min
        
        assert gap > 0
        
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        
        q.weight    = nn.Parameter(q.weight.detach(),   requires_grad=False)
        q.bias      = nn.Parameter(q.bias.detach(),     requires_grad=False)
        
        a = (nq - 1) / gap
        
        # First half equal to lines passing to (min+x, 1) and (min+x+1/a, 0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[: nq]  = -a
        q.bias[: nq]    = torch.from_numpy(a*min + np.arange(nq, 0, -1))            # b = 1 + a*(min+x)
        
        # Last half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:]   = a
        q.bias[nq:]     = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)               # b = 1 - a*(min+x)
        
        # First and last one as a horizontal straight line
        q.weight[0]     = q.weight[-1] = 0
        q.bias[0]       = q.bias[-1] = 1
        

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))
        

class TAPLoss (APLoss):
    """ Differentiable tie-aware AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1, simplified=False):
        APLoss.__init__(self, nq=nq, min=min, max=max)
        self.simplified = simplified

    def forward(self, x, label, qw=None, ret='1-mAP'):
        '''N: number of images;
           M: size of the descs;
           Q: number of bins (nq);
        '''
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        label = label.float()
        Np = label.sum(dim=-1, keepdim=True)

        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        c = q.sum(dim=-1)  # number of samples  N x Q = nbs on APLoss
        cp = (q * label.view(N, 1, M)).sum(dim=-1)  # N x Q number of correct samples = rec on APLoss
        C = c.cumsum(dim=-1)
        Cp = cp.cumsum(dim=-1)

        zeros = torch.zeros(N, 1).to(x.device)
        C_1d = torch.cat((zeros, C[:, :-1]), dim=-1)
        Cp_1d = torch.cat((zeros, Cp[:, :-1]), dim=-1)

        if self.simplified:
            aps = cp * (Cp_1d+Cp+1) / (C_1d+C+1) / Np
        else:
            eps = 1e-8
            ratio = (cp - 1).clamp(min=0) / ((c-1).clamp(min=0) + eps)
            aps = cp * (c * ratio + (Cp_1d + 1 - ratio * (C_1d + 1)) * torch.log((C + 1) / (C_1d + 1))) / (c + eps) / Np
        aps = aps.sum(dim=-1)

        assert aps.numel() == N

        if ret == '1-mAP':
            if qw is not None:
                aps *= qw  # query weights
            return 1 - aps.mean()
        elif ret == 'AP':
            assert qw is None
            return aps
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_tap'+('s' if self.simplified else ''): float(loss)}


class ArcMarginProduct(nn.Module):
    """
        Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        self.th = math.cos(math.pi - m)
        
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        # 1 * d
        input_n     = nn.functional.normalize(input)
        
        # d * n
        weight_n    = nn.functional.normalize(self.weight)
        
        # 1 * n
        cosine  = nn.functional.linear(input_n, weight_n)
        
        # 1 * n
        sine    = torch.sqrt( (1.0 - torch.pow(cosine, 2) ).clamp(0, 1) )

        # 1 * n        
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0,       phi,    cosine          )
            
        else:
            phi = torch.where(cosine > self.th, phi,    cosine - self.mm)
        
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(),    device='cuda')
        one_hot.scatter_(1,     label.view(-1, 1).long(),   1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'