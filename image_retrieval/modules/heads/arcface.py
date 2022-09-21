from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import partial

from core.utils.misc import ABN

from image_retrieval.modules.pools import POOLING_LAYERS

from image_retrieval.modules.losses import ArcMarginProduct, AddMarginProduct

from image_retrieval.modules.normalizations import NORMALIZATION_LAYERS, L2N


class globalHead(nn.Module):
    """ImageRetrievalHead for FPN
    Parameters
    """

    def __init__(self,  pooling=None, type=None, 
                        inp_dim=None, out_dim=None, 
                        s=30, margin=0.4,
                        do_withening=False, norm_act=ABN):

        super(globalHead, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # pooling
        if pooling["name"] == "GeMmp":
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
        else:
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])
            
        # type
        if type == 'arcface':
            self.final = ArcMarginProduct(inp_dim, out_dim, s=s, m=margin, easy_margin=False)
        elif type == 'cosface':
            self.final = AddMarginProduct(inp_dim, out_dim, s=s, m=margin)        
        else:
            self.final = nn.Linear(inp_dim, out_dim)    

        # Init
        self.reset_parameters()

    def reset_parameters(self):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                    nn.init.xavier_normal_(mod.weight, 0.1)

            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)


    def forward(self, x, label):

        # pool and normalize
        x = self.pool(x)

        # final
        x = self.final(x, label)

        # permute
        return x.permute(1, 0)
