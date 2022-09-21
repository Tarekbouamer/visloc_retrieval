import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.backbones.misc import ResidualBlock
 
from core.utils.misc import ABN, ALN
from image_retrieval.modules.cross_attention import Encoder
from timm.models.vision_transformer import trunc_normal_


class Next(nn.Module):
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
                 num_heads,
                 depth,
                                  
                 layer,
                 mlp_ratio=1,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 act=nn.GELU,
                 norm_act=ABN):

        super(Next, self).__init__()
        
        # 
        self.out_dim = structure[-1]
        
        # sampling:
        do_sample = [True] * (len(structure) - 1 )
        do_sample.append(False)
        
        # encoders 
        encoders = []
        for enc_id, (Ch, Nh, D, do)in enumerate(zip(structure, num_heads, depth, do_sample)):
        
            # Encoders 
            for d in range(D):
                if (d+1) < D:
                    out_dim = Ch
                    stride = 1 
                else:
                    try:
                        out_dim = structure[enc_id+1]
                        stride = 2 
                    except:
                        out_dim = structure[enc_id]
                        stride = 1
                        
                encoders.append(
                        # "Encoder{}{}".format(enc_id+1, d+1), 
                        Encoder(inp_dim=Ch,
                                out_dim=out_dim,
                                num_heads=Nh,
                                mlp_ratio=mlp_ratio,
                                stride = stride,
                                layer=layer, 
                                drop_path_rate=drop_path_rate,
                                act=act)
                        )
                            
        # Create module
        self.encoders = nn.ModuleList(encoders)

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

    def forward(self, x, H, W):
        """ 
        Args:
            x: torch.tensor
                input tensor [B, C, H, W] 
        """
        
        for enc in self.encoders:
            x, H, W = enc(x, H, W)
            
        return x, H, W


_NEXT = {
    "18":   {"structure": [128, 256, 384],  "num_heads": [2, 4, 6],     "depth": [1, 1, 1]  },
    "34":   {"structure": [128, 256, 512],  "num_heads": [4, 8, 16],    "depth": [2, 4, 2]  },
    "50":   {"structure": [256, 512],       "num_heads": [4, 8],        "depth": [4, 2]  }
}

__all__ = []

for name, params in _NEXT.items():
    net_name = "next" + name
    setattr(sys.modules[__name__], net_name, partial(Next, **params))
    __all__.append(net_name)
    