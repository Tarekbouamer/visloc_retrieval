
import torch
from torch.nn import Module, Dropout
import torch.nn as nn

from collections import OrderedDict

from image_retrieval.modules.projections import Subsample, LinearNorm, ConvNorm

from core.utils.misc import ABN


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()

        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values,):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
    
        return queried_values.contiguous()


class FullAttention(Module):
    
    def __init__(self, drop_rate=0.):
        super().__init__()
        
        # Drop
        self.dropout = Dropout(drop_rate)

    def forward(self, queries, keys, values):
        """ Multi-head scaled dot-product attention,
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """
                
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
  
        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5 
        
        A = torch.softmax(softmax_temp * QK, dim=2)
        
        # Drop out
        A = self.dropout(A)

        # Value
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class SelfAttention(nn.Module):
    def __init__(self, inp_dim, out_dim, num_heads, norm_act=ABN):
        super().__init__()
        
        self.inp_dim    = inp_dim
        self.out_dim    = out_dim
        self.num_heads  = num_heads

        self.mid_dim  = (self.inp_dim // self.num_heads )
        
        #
        self.f  = nn.Sequential(OrderedDict([
            ("conv",    nn.Conv2d(self.inp_dim, self.mid_dim, kernel_size=1, stride=1, bias=False)),
            ]))
        
        #
        self.g  = nn.Sequential(OrderedDict([
            ("conv",    nn.Conv2d(self.inp_dim, self.mid_dim, kernel_size=1, stride=1, bias=False)),
            ]))
        #
        self.h  = nn.Conv2d(self.inp_dim, self.mid_dim, kernel_size=1, stride=1, bias=False)
        self.v  = nn.Conv2d(self.mid_dim, self.out_dim, kernel_size=1, stride=1, bias=False)
        
        # softmax
        self.softmax = nn.Softmax(dim=-1)
        self.scale_dot_prod = self.mid_dim ** -0.5

    def forward(self, x):
        #
        B, C, H, W = x.shape
        
        # B mid_dim (HW)
        Q = self.f(x).view(B, self.mid_dim, -1)
        K = self.g(x).view(B, self.mid_dim, -1) 
        V = self.h(x).view(B, self.mid_dim, -1) 
        
        # B (HW) (HW)
        z = torch.bmm(Q.permute(0, 2, 1), K) 
        
        # Softmax the map 
        att = self.softmax(self.scale_dot_prod * z)
        
        # B N MC
        z = torch.bmm(att, V.permute(0, 2, 1)) 
        
        # B mid_dim H W
        z = z.permute(0, 2, 1).view(B, self.mid_dim, H, W) 

        # B C H W
        z = self.v(z)

        # apply
        z = z + x
        
        return z
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, inp_dim, out_dim, num_heads, resolution=(64,64), do_pos=False, norm_act=ABN):
        super().__init__()
        
        self.inp_dim    = inp_dim
        self.out_dim    = out_dim
        self.num_heads  = num_heads
        self.do_pos     = do_pos

        self.head_dim  = (self.inp_dim // self.num_heads )
    
        #
        self.f  = nn.Sequential(OrderedDict([
            ("conv",    nn.Conv2d(self.inp_dim, self.inp_dim, kernel_size=1, stride=1, bias=False)),
            ]))
        
        #
        self.g  = nn.Sequential(OrderedDict([
            ("conv",    nn.Conv2d(self.inp_dim, self.inp_dim, kernel_size=1, stride=1, bias=False)),
            ]))
        #
        self.h  = nn.Conv2d(self.inp_dim, self.inp_dim, kernel_size=1, stride=1, bias=False)
        self.v  = nn.Conv2d(self.inp_dim, self.out_dim, kernel_size=1, stride=1, bias=False)
        
        #
        self.self_attention = FullAttention()
        
        # merge
        self.merge          = nn.Linear(inp_dim, inp_dim, bias=False)
                             
            
    def forward(self, x):
        #
        B, C, H, W = x.shape
        
        # B mid_dim (HW)
        Q = self.f(x).view(B, -1, self.num_heads, self.head_dim)
        K = self.g(x).view(B, -1, self.num_heads, self.head_dim) 
        V = self.h(x).view(B, -1, self.num_heads, self.head_dim) 
        
        # self attention        
        z = self.self_attention(Q, K, V, size=(H, W))  # [N, L, (H, D)]
        
        # merge
        z = self.merge(z.view(B, -1, self.num_heads * self.head_dim))  # [N, L, C]
        
        # B C H W
        z = z.permute(0, 2, 1).view(B, self.inp_dim, H, W) 

        # Norm
        z = self.v(z)

        # add
        z = z + x
        
        return z


## TODO:  separated files uregently

