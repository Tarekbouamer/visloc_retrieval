import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn


from core.utils.misc import ABN, GlobalAvgPool2d

from .misc import ResidualBlock
from .util import try_index, CONV_PARAMS, BN_PARAMS



class Residual(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable or list of callable
        Function to create normalization / activation Module. If a list is passed it should have four elements, one for
        each module of the network
    classes : int
        If not `0` also include globalFeatures average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
        List of dilation factors for the four modules of the network, or `1` to ignore dilation
    dropout : list of float or None
        If present, specifies the amount of dropout to apply in the blocks of each of the four modules of the network
    caffe_mode : bool
        If `True`, use bias in the first convolution for compatibility with the Caffe pretrained models
    """

    def __init__(self,
                 structure,
                 bottleneck,
                 
                 norm_act=ABN,
                 config=None,
                 classes=0,
                 dilation=1,
                 dropout=None,
                 drop_path_rate=0.0,
                 caffe_mode=False):
        super(Residual, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.dilation = dilation
        self.dropout = dropout
        self.caffe_mode = caffe_mode


        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")
        
        # 
        block_dprs = [x.tolist() for x in torch.linspace(0.0, drop_path_rate, sum(structure)).split(structure)]
        
        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=caffe_mode)),
            ("bn1", try_index(norm_act, 0)(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 96)
        else:
            channels = (64, 64)
            
        for mod_id, num in enumerate(structure):
            mod_dropout = None
            if self.dropout is not None:
                if self.dropout[mod_id] is not None:
                    mod_dropout = partial(nn.Dropout, p=self.dropout[mod_id])
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                
                blocks.append((
                    "block%d" % (block_id + 1),
                    ResidualBlock(in_channels, channels, norm_act=try_index(norm_act, mod_id),
                                  stride=stride, dilation=dil, 
                                  drop_path_rate=block_dprs[mod_id][block_id], 
                                  dropout=mod_dropout)
                ))

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]
        
        # Out dim
        self.out_dim = in_channels
        
        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))
            
            self.out_dim = classes


    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    
    def forward(self, x):
        outs = OrderedDict()

        x = self.mod1(x)
        x = self.mod2(x)
        x = self.mod3(x)
        x = self.mod4(x)

        return x


_NETS = {
    "18": {"structure": [2, 2],     "bottleneck": False},
    "34": {"structure": [3, 4],     "bottleneck": False},
    "50": {"structure": [3, 4, 4],  "bottleneck": False},
    
}

__all__ = []

for name, params in _NETS.items():
    net_name = "residual" + name
    setattr(sys.modules[__name__], net_name, partial(Residual, **params))
    __all__.append(net_name)
