from collections import OrderedDict

import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as functional

from functools import partial

from core.utils.misc import ABN, ALN

from image_retrieval.modules.pools import POOLING_LAYERS
from image_retrieval.modules.normalizations import NORMALIZATION_LAYERS, L2N
from image_retrieval.modules.attention import SelfAttention, MultiHeadSelfAttention
from image_retrieval.modules.transformer import Encoder


class globalHead(nn.Module):
    """ Global Image Descriptor head, 
    consists of two main operations (1) pooling and (2) withening, each followed by L2 normalization.
    
    Args:
        pooling: struct
                pooling paramaters {name, coefficient...} 
        withening: boolean 
                weather the final the descriptor is withened or not. 
        inp_dim: int
                the initial descriptor size, after performing pooling opertion.    
        out_dim: int 
                the final descriptor size, if out_dim < inp_dim, model reduction throught linear layer.    
    """

    def __init__(self, inp_dim=None, out_dim=None,
                 
                 num_encs=1,
                 
                 attention=None,
                 num_heads=8,
                 resolution=(64,64),
                 
                 pooling=None,
                 do_withening=False, 
                 norm_act=ABN):

        super(globalHead, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attention = attention
        self.num_encs = num_encs
        
        print(self.attention)

        self.do_withening = do_withening
        
        # Transformer
        if self.attention ==  "transformer":
            encoders = []
            
            for enc_id in range(self.num_encs):
                encoders.append((
                        "encoder%d" % (enc_id + 1),
                        Encoder( d_model=self.inp_dim, num_heads=self.num_heads, attention="full")
                    ))
                
            # Create module
            self.add_module("encoders", nn.Sequential(OrderedDict(encoders)))
        
        # Attention
        elif self.attention in  ["self", "multi", "pos"]:
            
            if self.attention == 'self':
                self.encoders = SelfAttention(inp_dim=inp_dim, out_dim=out_dim, num_heads=num_heads, norm_act=norm_act)
            
            elif self.attention == 'multi':
                self.encoders = MultiHeadSelfAttention(inp_dim=inp_dim, out_dim=out_dim, num_heads=num_heads, norm_act=norm_act)

            elif attention == 'pos':
                self.encoders = MultiHeadSelfAttention(inp_dim=inp_dim, out_dim=out_dim, num_heads=num_heads, 
                                                       resolution=resolution,
                                                       do_pos=True,
                                                       norm_act=norm_act) 
        # Pooling
        if pooling["name"] == "GeMmp":
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
        else:
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])

        # Whitening
        if self.do_withening:
            self.whiten = nn.Linear(inp_dim, out_dim, bias=True)
        
        # Init 
        self.reset_parameters()

    def reset_parameters(self):
        
        for name, mod in self.named_modules():
            
            if isinstance(mod,  nn.Linear):
                nn.init.xavier_normal_(mod.weight, 0.01)

            elif isinstance(mod,    nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, 0.01)
                
            elif isinstance(mod,    ALN) or isinstance(mod, nn.LayerNorm):
                nn.init.constant_(mod.weight, 1.0)
            
            elif isinstance(mod,    ABN) or isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1.0)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)


    def forward(self, x, do_whitening=True):
        """ 
            Parameters
        """
        
        # encoder 
        if self.attention is not None:
            x = self.encoders(x)
        
        # pooling 
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)
        
        # whithen and normalize
        if do_whitening and self.do_withening: 
            x = self.whiten(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # permute
        return x.permute(1, 0)
