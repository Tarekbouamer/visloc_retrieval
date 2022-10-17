from functools import partial

import torch
import torch.nn as nn


from image_retrieval.modules.pools import GeM
from image_retrieval.modules.heads.head import register_head, model_entrypoint, is_model
from .base import Head

_cfg = {"name": 'Gem',  "p": 3}
 
@register_head
def gem_linear(inp_dim, out_dim, p=None,  **kwargs):
    return GemHead(inp_dim, out_dim, p=p, layer="linear")
    

@register_head
def gem_conv(inp_dim, out_dim, p=None,  **kwargs):  
    return GemHead(inp_dim, out_dim, p=p, layer="conv")    

    
class GemHead(Head):
    """ GemHead
    """
    def __init__(self, inp_dim, out_dim, layer="linear", **kwargs):
        super(GemHead, self).__init__(inp_dim, out_dim)

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        #
        layer   = kwargs.pop("layer", "linear")
        p       = kwargs.pop("p", 3.0)
               
        # layer
        layer = nn.Linear if layer=="linear" else partial(nn.Conv2d, kernel_size=(1,1), padding=0)

        # pooling
        self.pool = GeM(p=3.0, eps=0.0)

        # whitening
        self.whiten = layer(inp_dim, out_dim, bias=True)
             
        # init 
        self.reset_parameters()
    
    def forward(self, x, do_whitening=True):
        """ 
        Args:
                x: torch.tensor
                    input tensor [B, C, H, W] 
                do_whitening: boolean 
                    do or skip whithening (Note: used to initilize the linear layer using PCA) 
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
        **kwargs):
    """Create a model
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # 
    # if not is_model(model_name):
    #     raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
   
    model = create_fn(**kwargs)

    return model
    
if __name__ == '__main__':
    
    create_fn = create_head("gem_linear", pretrained=True)
    print(create_fn)      