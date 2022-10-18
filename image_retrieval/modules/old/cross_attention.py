from ast import Not
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional 

from core.utils.misc import ABN

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
     
class Subsample(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride * stride

    def forward(self, x):
        
        B, N, C = x.shape
                
        x = x.view(B, N, C)[:, ::self.stride ]
        
        return x.reshape(B, -1, C)    
    
    
class ConvNorm(nn.Sequential):
    def __init__(self, inp_dim, out_dim=None, bias=False, act=nn.GELU):
        super().__init__()
    
        self.add_module('conv', nn.Conv2d(inp_dim,   out_dim,  kernel_size=1, bias=bias))
        self.add_module('bn',   nn.BatchNorm2d(out_dim)                     )        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  
        return x


class LinearNorm(nn.Sequential):
  
    def __init__(self, inp_dim, out_dim=None,  bias=False):
        super().__init__()
        
        self.add_module('conv', nn.Linear(inp_dim,   out_dim,   bias=bias)  )
        self.add_module('bn',   nn.LayerNorm(out_dim)                       )        
    
    def forward(self, x):
        x = self.conv(x)
        x= self.bn(x)
        return x
    
     
class ResidualBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, residuals, norm_act=ABN, init_values=1e-4, drop_path_rate=0):
        super().__init__()
        
        need_proj_conv = inp_dim != out_dim
          
        # norm
        self.norm = nn.LayerNorm(inp_dim)

        self.residuals = residuals
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # weights
        self.gamma  = nn.Parameter(init_values * torch.ones((out_dim)), requires_grad=True)
               
    def forward(self, x, H=None, W=None):

        z, H, W = self.residuals(self.norm(x), H, W)

        x = x + self.drop_path(self.gamma * z)        
            
        return x, H, W

class MLP(nn.Module):
    """ MLP Basic
    """
    def __init__(self, dim=None, mlp_ratio=2, act=nn.GELU, drop_rate=0., bias=False):
        super().__init__()
        
        mlp_dim = int(dim * mlp_ratio)

        drop_probs = to_2tuple(drop_rate)
        
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = act()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x) 
        
        return x, H, W


class ConvMlp(nn.Module):
    """ MLP convs that keeps spatial
    """
    def __init__( self, dim, mlp_ratio=2, act=nn.GELU, drop_rate=0., bias=False):
        super().__init__()
        
        self.convs = nn.Sequential(OrderedDict(
            [
                ('fc1',     nn.Conv2d(dim,  dim * mlp_ratio,    kernel_size=1,  bias=bias)  ),
                ('bn1',     nn.BatchNorm2d( dim * mlp_ratio )                               ),
                ('act',     act()                                                           ),
                ('fc2',     nn.Conv2d(dim * mlp_ratio,  dim, kernel_size=1, bias=bias)      ),
                ('bn2',     nn.BatchNorm2d( dim )                                           )
                ])
            )
    def forward(self, x):
        x = self.convs(x)
        return x
    
              
class Attention(nn.Module):
  
    def __init__(self, inp_dim, out_dim, num_heads=None,
                 bias=False, layer='conv', 
                 drop_rate=0.,
                 act=nn.GELU):
        super().__init__()
        
        self.num_heads  =  num_heads
        
        self.temperature    = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv        = nn.Linear(inp_dim, inp_dim * 3, bias=bias)
        self.attn_drop  = nn.Dropout(drop_rate)

        self.proj       = nn.Linear(inp_dim, inp_dim, bias=bias)
        self.proj_drop  = nn.Dropout(drop_rate)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, H, W
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
   
    
class Projection(nn.Module):
    """ Projection 
    """

    def __init__(self,
                 inp_dim,
                 out_dim,
                 kernel=3,
                 stride=1,
                 norm_act=ABN,
                 act=nn.GELU,):
        super(Projection, self).__init__()

        padding =  kernel // 2
        stride = 2  if inp_dim != out_dim else 1
        
        need_proj_conv = inp_dim != out_dim 
        
        # convs
        layers = [
            ("conv1",   nn.Conv2d(inp_dim, out_dim, kernel, stride=stride,  padding=padding, bias=False) ),
            ("bn1",     nn.BatchNorm2d(out_dim)),
            ("act1",    act()),
            ("conv2",   nn.Conv2d(out_dim, out_dim, kernel, stride=1,       padding=padding, bias=False) ),
            ]
        
        self.convs = nn.Sequential(OrderedDict(layers))
         
                                
    def forward(self, x, H, W):
        # reshape
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  
              
        #  
        x = self.convs(x)
        
        # reshape again
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        return x, H, W
        
                           
class Encoder(nn.Module):
    
    def __init__(self, inp_dim, out_dim,  num_heads=4, mlp_ratio=2, stride=1,
                 drop_path_rate=0., drop_rate=0., init_values=0.05,
                 layer="conv", act=nn.GELU, norm_act=ABN):
        super().__init__()
        
        # linear or conv layers
        self.use_linear = layer=="linear"
         
        # A) Attention Residual Block 
        attention = Attention(inp_dim=inp_dim, out_dim=inp_dim, num_heads=num_heads,
                              drop_rate=drop_rate, 
                              layer=layer, act=act)
        
        self.Attention_Block = ResidualBlock(residuals=attention, 
                                             inp_dim=inp_dim,
                                             out_dim=inp_dim, 
                                             drop_path_rate=drop_path_rate)
                                       
        # B)  MLP Residual Block
        mlp_layer = MLP if layer=="linear" else  ConvMlp
        
        mlp = mlp_layer(dim=inp_dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate, bias=False, act=act) 

        self.MLP_Block = ResidualBlock(residuals=mlp,
                                       inp_dim=inp_dim, 
                                       out_dim=inp_dim,
                                       drop_path_rate=drop_path_rate)   
        
        # C ) Projection     
        proj =  Projection(inp_dim, out_dim, 
                                act=act)

        self.Projection_Block = ResidualBlock(residuals=proj,
                                              inp_dim=inp_dim, 
                                              out_dim=out_dim, 
                                              drop_path_rate=drop_path_rate)
            

    def forward(self, x, H, W):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """

        # Attention 
        x, _, _ = self.Attention_Block(x)
        
        # MLP
        x, _, _ = self.MLP_Block(x)   
        
        # Projection
            
        x, H, W = self.Projection_Block(x, H, W)

        
        return x, H, W
    
