# from collections import OrderedDict

# import torch
# from torch._C import set_flush_denormal
# import torch.nn as nn
# import torch.nn.functional as functional

# from functools import partial

# from core.utils.misc import ABN, ALN

# from image_retrieval.modules.pools import POOLING_LAYERS
# from image_retrieval.modules.normalizations import NORMALIZATION_LAYERS, L2N
# from image_retrieval.modules.attention import SelfAttention, MultiHeadSelfAttention


# class AttentionHead(nn.Module):
#     """
#         TransformerHead
#             Parameters
#     """

#     def __init__(self, inp_dim=None, out_dim=None,
#                  attention='self',
#                  num_heads=8,
#                  resolution=(64,64),
#                  pooling=None,
#                  do_withening=False, 
#                  norm_act=ABN):

#         super(AttentionHead, self).__init__()

#         self.inp_dim = inp_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads

#         self.do_withening = do_withening
        
#         # Attention
#         if attention == 'self':
#             self.attention = SelfAttention(inp_dim=inp_dim, 
#                                            out_dim=out_dim,
#                                            num_heads=num_heads,
#                                            norm_act=norm_act)
#         elif attention == 'multi':
#             self.attention = MultiHeadSelfAttention(inp_dim=inp_dim, 
#                                                     out_dim=out_dim,
#                                                     num_heads=num_heads,
#                                                     norm_act=norm_act)

#         elif attention == 'pos':
#             self.attention = MultiHeadSelfAttention(inp_dim=inp_dim, 
#                                                     out_dim=out_dim,
#                                                     num_heads=num_heads,
#                                                     resolution=resolution,
#                                                     do_pos=True,
#                                                     norm_act=norm_act)            
#         # pooling
#         if pooling["name"] == "GeMmp":
#             self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
#         else:
#             self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])

#         # Whitening
#         if self.do_withening:
#             self.whiten = nn.Linear(inp_dim, out_dim, bias=True)
        
#         # init 
#         self.reset_parameters()

#     def reset_parameters(self):
        
#         # gain = nn.init.calculate_gain(self.attention.f.bn.activation, self.attention.f.bn.activation_param)

#         for name, mod in self.named_modules():
            
#             if isinstance(mod,      nn.Linear):
#                 nn.init.xavier_normal_(mod.weight, 0.1)

#             if isinstance(mod,      nn.Conv2d):
#                 nn.init.xavier_normal_(mod.weight, 0.01)
                    
#             elif isinstance(mod,    ABN) or isinstance(mod,    ALN):
#                 nn.init.constant_(mod.weight, 1.0)

#             if hasattr(mod, "bias") and mod.bias is not None:
#                 nn.init.constant_(mod.bias, 0.)


#     def forward(self, x, do_whitening=True):
#         """ 
#             Parameters
#         """
        
#         # attention 
#         # if do_whitening:
#         x = self.attention(x)
        
#         # pooling 
#         x = self.pool(x)
#         x = x.squeeze(-1).squeeze(-1)
#         x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)
        
#         # whithen and normalize
#         if do_whitening and self.do_withening: 
#             x = self.whiten(x)
#             x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

#         # permute
#         return x.permute(1, 0)
