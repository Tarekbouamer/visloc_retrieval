from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional


from core.utils.misc import ABN

from image_retrieval.modules.how import L2Attention, SmoothingAvgPooling , DimReduction


class HowHead(nn.Module):
    """
      ImageRetrievalHead for Learning and aggregating deep local descriptors for instance-level recognition
      Parameters
    """

    def __init__(self, inp_dim=None, out_dim=None, kernel_size=3, do_withening=False, norm_act=ABN):

        super(HowHead, self).__init__()

        self.inp_dim        = inp_dim
        self.out_dim        = out_dim
        self.kernel_size    = kernel_size

        self.pad = kernel_size // 2

        # do withening
        self.do_withening = do_withening

        # attention
        self.attention  = L2Attention() 

        # whiten
        self.whiten     = nn.Conv2d(inp_dim, out_dim, (1, 1), padding=0, bias=True)
 
        # init  
        self.reset_parameters()

    def reset_parameters(self):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                    nn.init.xavier_normal_(mod.weight, 0.1)

            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)


    def forward(self, x, do_whitening=True):
        """ 
            ImageRetrievalHead
            Parameters
        """

        # attention 
        att     = self.attention(x)
        n_att   = att / att.max()

        # smoothing
        x = functional.avg_pool2d(  x,
                                    kernel_size=(self.kernel_size, self.kernel_size), 
                                    stride=1, 
                                    padding=self.pad,
                                    count_include_pad=False)
        
        if do_whitening:
            x = self.whiten(x)        

        # compute
        desc = (x * n_att).sum((-2, -1))
        
        # normalize
        desc = functional.normalize(desc, p=2, dim=-1)
                
        # Normalize max weight to 1
        return desc.permute(1, 0)
