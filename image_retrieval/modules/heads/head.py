from functools import partial

import torch
import torch.nn as nn


from image_retrieval.modules.pools import POOLING_LAYERS
from image_retrieval.models.registry import register_model


gem_cfg = {"name": "GeM", "params": {"p":3, "eps": 1e-6}}

# TODO: chamge the register_model to register_head
# add alternatives 
 
@register_model
def gem_linear(inp_dim, out_dim, p=None,  **kwargs):
    
    if p is not None:
        gem_cfg["params"]["p"] = p
    
    return RetrievalHead(inp_dim, out_dim, pooling=gem_cfg, layer="linear")
    

@register_model
def gem_conv(inp_dim, out_dim, p,  **kwargs):
    if p is not None:
        gem_cfg["params"]["p"] = p
            
    return RetrievalHead(inp_dim, out_dim, pooling=gem_cfg, layer="conv")    



    
class RetrievalHead(nn.Module):
    
    """ Global Image Descriptor Head 
    Consists of two main operations (1) pooling and (2) withening, each followed by L2 normalization.
    
    Args:
        pooling: struct
                pooling paramaters {name, coefficient...} 
        withening: boolean 
                weather the final the descriptor is withened or not. 
        inp_dim: int
                the initial descriptor size, after performing pooling opertion.    
        out_dim: int 
                the final descriptor size, if out_dim < inp_dim, model reduction throught linear layer.
        norm_act: nn.module
                the nomlization layer and activation function used for this head.     
    """

    def __init__(self, inp_dim, out_dim, pooling=None, layer="linear"):
        super(RetrievalHead, self).__init__()

        self.inp_dim    = inp_dim
        self.out_dim    = out_dim
               
        # layer
        layer = nn.Linear if layer=="linear" else partial(nn.Conv2d, kernel_size=(1,1), padding=0)

        # pooling
        if pooling["name"] == "GeMmp":
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
        else:
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])

        # whitening
        self.whiten = layer(inp_dim, out_dim, bias=True)
             
        # init 
        self.reset_parameters()

    def reset_parameters(self):
        for name, mod in self.named_modules():
            if isinstance(mod,  nn.Linear) or isinstance(mod,  nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, 1.0)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)
     
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
    
    

# class _RetrievalHead(nn.Module):
    
#     """ Global Image Descriptor Head 
#     Consists of two main operations (1) pooling and (2) withening, each followed by L2 normalization.
    
#     Args:
#         pooling: struct
#                 pooling paramaters {name, coefficient...} 
#         withening: boolean 
#                 weather the final the descriptor is withened or not. 
#         inp_dim: int
#                 the initial descriptor size, after performing pooling opertion.    
#         out_dim: int 
#                 the final descriptor size, if out_dim < inp_dim, model reduction throught linear layer.
#         norm_act: nn.module
#                 the nomlization layer and activation function used for this head.     
#     """

#     def __init__(self, 
#                  inp_dim, 
#                  out_dim,
#                  pooling=None,
#                  layer="linear"):

#         super(RetrievalHead, self).__init__()

#         self.inp_dim    = inp_dim
#         self.out_dim = out_dim
       
#         # Layer
#         layer = nn.Linear if layer=="linear" else partial(nn.Conv2d, kernel_size=(1,1), padding=0)

#         # Pooling
#         if pooling["name"] == "GeMmp":
#             self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
#         else:
#             self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])

#         # Whitening
#         self.whiten = layer(inp_dim, out_dim, bias=True)
             
#         # Init 
#         self.reset_parameters()

#     def reset_parameters(self):
        
#         for name, mod in self.named_modules():
            
#             if isinstance(mod,  nn.Linear):
#                 nn.init.xavier_normal_(mod.weight, 0.01)
                
#             elif isinstance(mod,    ALN) or isinstance(mod, nn.LayerNorm):
#                 nn.init.constant_(mod.weight, 1.0)

#             if hasattr(mod, "bias") and mod.bias is not None:
#                 nn.init.constant_(mod.bias, 0.)
          
#     def get_locations(self, H, W):
#         # B, L, 2
#         grid_y, grid_x  = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#         grid            = torch.stack((grid_x, grid_y), 2).reshape(-1, 2)
#         locs            = grid.unsqueeze(0)
        
#         return locs
    
#     def random_sampler(self, L, K=None):
#         K           = min(K, L) if K is not None else L
#         indices     = torch.randperm(L)[:K]  
#         return indices 
          
#     def forward(self, x, scales=[1.0], do_whitening=True):
#         """ 
#         Args:
#                 x: torch.tensor
#                     input tensor [B, C, H, W] 
#                 do_whitening: boolean 
#                     do or skip whithening (Note: used to initilize the linear layer using PCA) 
#         """
#         descs = []
                    
#         for it, (s_i, x_i) in enumerate(zip(scales, x)):

#             # pooling        
#             x_i = self.pool(x_i)
#             x_i = x_i.squeeze(-1).squeeze(-1)
#             x_i = nn.functional.normalize(x_i, dim=-1, p=2, eps=1e-6)

#             # whithen and normalize
#             if do_whitening: 
#                 x_i = self.whiten(x_i)
#                 x_i = nn.functional.normalize(x_i, dim=-1, p=2, eps=1e-6)
                
#             # append
#             descs.append(x_i)
            
#         return descs            
    
#     def forward_locals(self, x, scales, do_whitening):
#         """ 
#             To a list of local descriptors for every scale.
#         """
#         descs, locs = [], []
        
#         for it, x_i in enumerate(x):

#             # do_whitening
#             if do_whitening and self.do_withening: 
#                 x_i = self.local_whiten(x_i)
#                 x_i = nn.functional.normalize(x_i, dim=-1, p=2, eps=1e-6)
                
#             # resolution             
#             B, C, H, W  = x_i.shape
                    
#             #             
#             descs_i     = x_i.reshape(B, C, -1).permute(0, 2, 1)
#             locs_i      = self.get_locations(H, W)
                        
#             # Append
#             descs.append(descs_i)
#             locs.append(locs_i)
                    
#         return descs, locs