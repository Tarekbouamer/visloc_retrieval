from functools import partial

import torch.nn as nn


from image_retrieval.modules.pools import GeM
from .registry import register_head, model_entrypoint, is_model

from .base import Head


@register_head
def gem_linear(inp_dim, out_dim, **kwargs):
    return GemHead(inp_dim, out_dim, layer="linear", **kwargs)
    

@register_head
def gem_conv(inp_dim, out_dim, **kwargs):  
    return GemHead(inp_dim, out_dim, layer="conv", **kwargs)    


    
class GemHead(Head):
    """ Generalized Mean Pooling Head
    
    references:
                https://arxiv.org/pdf/1711.02512.pdf
                https://github.com/filipradenovic/cnnimageretrieval-pytorch     
    Args:

        inp_dim:    int         the input features size  
        out_dim:    int         the final descriptor size
        layer:      str         whiten layer linear/Conv2d 
    
        kwargs      dict        GeM pooling coefficient p=3.0    
                
    """
    def __init__(self, inp_dim, out_dim, layer="linear", **kwargs):
        super(GemHead, self).__init__(inp_dim, out_dim)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        #
        p       = kwargs.pop("p", 3.0)
               
        # layer
        layer = nn.Linear if layer=="linear" else partial(nn.Conv2d, kernel_size=(1,1), padding=0)

        # pooling
        self.pool = GeM(p=p)

        # whitening
        self.whiten = layer(inp_dim, out_dim, bias=True)
             
        # init 
        self.reset_parameters()
    
    def forward(self, x, do_whitening=True):
        """ 
        Args:
                x:              torch.Tensor    input tensor [B, C, H, W] 
                do_whitening:   boolean         do or skip whithening 
        """
                
        # pooling        
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # whithen 
        if do_whitening: 
            x = self.whiten(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)
            
        return x     


def create_head(
        model_name,
        inp_dim, 
        out_dim,
        **kwargs):
    """Create a head 
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
   
    model = create_fn(inp_dim, out_dim,**kwargs)

    return model
    
    
# test
if __name__ == '__main__':
    
    create_fn = create_head("gem_conv", 100, 10, p=1.22)
    print(create_fn)      