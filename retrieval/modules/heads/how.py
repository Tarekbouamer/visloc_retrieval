from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional


from .registry import register_head
from .factory import create_head

from .base import BaseHead


# how  list of implementations 

@register_head
def how(inp_dim, out_dim, kernel_size=3, **kwargs):
    return HowHead(inp_dim, out_dim, kernel_size, **kwargs)
 
    
# 
class L2Attention(nn.Module):
    """ attention as L2-norm of local descriptors   """
    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)

# 
class SmoothingAvgPooling(nn.Module):
    """ Average pooling that smoothens the feature map   """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        pad = self.kernel_size // 2
        return functional.avg_pool2d(x, self.kernel_size, stride=1, padding=pad, count_include_pad=False)
    
    def __repr__(self):
          return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'

# 
class ConvDimReduction(nn.Conv2d):
    """ Dimensionality reduction as a convolutional layer """
    def __init__(self, inp_dim, out_dim):
        super().__init__(inp_dim, out_dim, (1, 1), padding=0, bias=True)
    
#    
class HowHead(BaseHead):
    """ 
        How head for Learning and aggregating deep local descriptors for instance-level recognition
    
    references:
                https://arxiv.org/abs/2007.13172
                https://github.com/Tarekbouamer/Visloc/blob/master/image_retrieval/modules/heads/how_head.py    
                
    """
    def __init__(self, inp_dim, out_dim=128, kernel_size=3, **kwargs):
        super(HowHead, self).__init__(inp_dim, out_dim)

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        # attention
        self.attention = L2Attention()
        
        # pool
        self.pool = SmoothingAvgPooling(kernel_size=kernel_size)      # specifiy the kernel size
        
        # whiten
        self.whiten = nn.Conv2d(inp_dim, out_dim, kernel_size=(1,1), padding=0, bias=True)
             
        # init 
        self.reset_parameters()
        
        # not trainable  
        for param in self.whiten.parameters():
            param.requires_grad = False
    
    def forward(self, x, do_whitening=True):
        
        # attention 
        attn = self.attention(x)
        
        # pool and reduction  
        x = self.pool(x)
        
        if do_whitening:
            x = self.whiten(x)
        
        #
        preds = {
            'features': x ,
            'attns': attn
        }
            
        return preds



    
    
# test
if __name__ == '__main__':
    
    create_fn = create_head("how", 1024, 128, p=1.22)
   
    print(create_fn)      