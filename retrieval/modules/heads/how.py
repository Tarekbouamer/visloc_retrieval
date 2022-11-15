from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional


from retrieval.modules.pools import GeM
from .registry import register_head, model_entrypoint, is_model

from .base import Head

# how  list of implementations 

@register_head
def how(inp_dim, out_dim, kernel_size=3, **kwargs):
    return HowHead(inp_dim, out_dim, kernel_size, **kwargs)
    

# 
def smoothing_avg_pooling(feats, kernel_size):
    """
        Smoothing average pooling
            :param torch.Tensor feats: Feature map
            :param int kernel_size: kernel size of pooling
            :return torch.Tensor: Smoothend feature map
    """
    pad = kernel_size // 2
    return functional.avg_pool2d(feats, (kernel_size, kernel_size), stride=1, padding=pad, count_include_pad=False)


# 
class L2Attention(nn.Module):
    """ 
        attention as L2-norm of local descriptors
    """

    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)
        # return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)

# 
class SmoothingAvgPooling(nn.Module):
    """
        Average pooling that smoothens the feature map, 
        keeping its size
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return smoothing_avg_pooling(x, kernel_size=self.kernel_size)

# 
class ConvDimReduction(nn.Conv2d):
    """
        Dimensionality reduction as a convolutional layer
        :param int input_dim: Network out_channels
        :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, (1, 1), padding=0, bias=True)
    
#    
class HowHead(Head):
    """ 
        How head for Learning and aggregating deep local descriptors for instance-level recognition
    
    references:
                https://arxiv.org/abs/2007.13172
                https://github.com/Tarekbouamer/Visloc/blob/master/image_retrieval/modules/heads/how_head.py    
    Args:

                
    """
    def __init__(self, inp_dim, out_dim=128, kernel_size=3, **kwargs):
        super(HowHead, self).__init__(inp_dim, out_dim)

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        # 
        self.attention = L2Attention()
        
        # pool
        self.pool = SmoothingAvgPooling(kernel_size=kernel_size)      # specifiy the kernel size
        
        # whiten
        self.whiten = nn.Conv2d(inp_dim, out_dim, kernel_size=(1,1), padding=0, bias=True)
             
        # init 
        self.reset_parameters()
    
    def forward(self, x, do_whitening=True):
        """ 
        Args:
                x:              torch.Tensor    input tensor [B, C, H, W] 
                do_whitening:   boolean         do or skip whithening 
        """
        
        # attention 
        attn = self.attention(x)
        
        # pool and reduction  
        x = self.pool(x)
        
        if do_whitening:
            x = self.whiten(x)
        
        #
        preds = {
            'feats': x ,
            'attns': attn
        }
            
        return preds


def create_head(
        head_name,
        inp_dim, 
        out_dim,
        **kwargs):
    """Create a head 
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not is_model(head_name):
        raise RuntimeError('Unknown model (%s)' % head_name)

    create_fn = model_entrypoint(head_name)
   
    head = create_fn(inp_dim, out_dim,**kwargs)

    return head
    
    
# test
if __name__ == '__main__':
    
    create_fn = create_head("how", 1024, 128, p=1.22)
   
    print(create_fn)      