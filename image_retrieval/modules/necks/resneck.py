import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional


from core.utils.misc import ABN, ALN
from image_retrieval.modules.self_attention import Encoder, SubSample
from timm.models.vision_transformer import trunc_normal_


class Neck(nn.Module):
    """ Features Attention Neck,
    
    Consists of a sequence of encoders based on attention mechanism and set of MLPs to be performed on the backbone features.
    
    Args:
        inp_dim: int 
                input feature size
        out_dim: int 
                desired out feature size, possible reduction 
        num_encs: int
                number of encoders, depth of the neck module 
        attention: str
                type off attention mechanism or module   
        num_heads: int  
                for multi head attention.
        resolution: tuple
                maximum spatial size of the input feature tensor (Hmax, Wmax)    
        norm_act: nn.module  
                the nomlization layer and activation function used for this head..    
    """

    def __init__(self, 
                 structure, 
                 key_dim,
                 num_heads,
                 depth,
                                  
                 layer,
                 att_ratio=1,
                 mlp_ratio=1,
                 drop_rate=0.,
                 act=nn.GELU):

        super(Neck, self).__init__()
        
        # 
        self.out_dim = structure[-1]
        
        # sampling:
        do_sample = [True] * (len(structure) - 1 )
        do_sample.append(False)
        
        # encoders 
        encoders = []
        for enc_id, (Ch, Nh, D, do)in enumerate(zip(structure , num_heads, depth, do_sample)):
        
            # Encoders 
            for d in range(D):
                encoders.append((
                        "encoder {} {}".format(enc_id+1, d+1), 
                        Encoder(inp_dim=Ch, key_dim=key_dim, 
                                num_heads=Nh, att_ratio=att_ratio, 
                                mlp_ratio=mlp_ratio,layer=layer, 
                                drop_rate=drop_rate,act=act))
                                )
            # Sub_Sample
            if do:
                encoders.append((
                        "sub_sample %d" % (enc_id + 1), 
                        SubSample(inp_dim=Ch, out_dim=structure[enc_id + 1],key_dim=key_dim, 
                                  num_heads=Nh,stride=2, att_ratio=att_ratio, 
                                  mlp_ratio=mlp_ratio,  layer=layer, 
                                  drop_rate=drop_rate,act=act
                                                                    ))
                                )
        
        # Create module
        self.add_module("encoders", nn.Sequential(OrderedDict(encoders)))

        # init module
        self.reset_parameters()

    def reset_parameters(self):
        
        for name, mod in self.named_modules():
            # 
            if isinstance(mod,  nn.Linear):
                trunc_normal_(mod.weight, 0.02)

            elif isinstance(mod,    nn.Conv2d):
                trunc_normal_(mod.weight, 0.02)
            #    
            elif isinstance(mod,    ALN) or isinstance(mod, nn.LayerNorm):
                nn.init.constant_(mod.weight, 1.0)

            elif isinstance(mod,    ABN) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm1d):
                nn.init.constant_(mod.weight, 1.0)

            #
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0)

    def forward(self, x):
        """ 
        Args:
            x: torch.tensor
                input tensor [B, C, H, W] 
        """
        
        x = self.encoders(x)
        
        return x


_NECKS = {
    "18":   {"structure": [256, 512],   "key_dim":16,   "num_heads": [4, 8], "depth": [2, 2]    },
    "50":   {"structure": [512, 1024],  "key_dim":32,   "num_heads": [4, 8], "depth": [1, 1]    }
}

__all__ = []

for name, params in _NECKS.items():
    net_name = "neck" + name
    setattr(sys.modules[__name__], net_name, partial(Neck, **params))
    __all__.append(net_name)
    