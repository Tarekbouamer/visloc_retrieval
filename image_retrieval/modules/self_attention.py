from collections import OrderedDict

import torch
import torch.nn as nn

     
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
        self.add_module('bn',   nn.BatchNorm1d(out_dim)                     )        
    
    def forward(self, x):        
        x = self.conv(x)
        x= self.bn(x.flatten(0, 1)).reshape_as(x)
        return x
    
     
class ResidualBlock(nn.Module):
    def __init__(self, residuals, drop=0):
        super().__init__()
        
        self.residuals = residuals
        self.drop = drop
        
    def forward(self, x): 
        if self.drop > 0:
            return x + self.residuals(x) * torch.rand(x.size(0), 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.residuals(x)

class MLP(nn.Module):
    """ MLP Basic
    """
    def __init__(self, dim=None, mlp_ratio=2, act=nn.GELU, drop_rate=0., bias=False):
        super().__init__()

        self.fc1 = nn.Linear(dim,  dim * mlp_ratio,     bias=bias)
        self.bn1 = nn.BatchNorm1d(dim * mlp_ratio)
        
        self.act = act()                                          
        
        self.fc2 = nn.Linear(dim * mlp_ratio,   dim,    bias=bias) 
        self.bn2 = nn.BatchNorm1d(dim)
    

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x.flatten(0, 1)).reshape_as(x)
        
        x = self.act(x)
        
        x = self.fc2(x)
        x = self.bn2(x.flatten(0, 1)).reshape_as(x)
        
        return x


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
  
    def __init__(self, inp_dim, key_dim=None, 
                 num_heads=None, att_ratio=4, 
                 bias=False, drop_rate=0.5, 
                 layer='conv', act=nn.GELU):
        super().__init__()
        
        self.inp_dim    = inp_dim
        self.num_heads  =  num_heads
        self.key_dim    = key_dim
        
        # individual dims 
        self.q_dim = key_dim
        self.k_dim = key_dim
        self.v_dim = int(att_ratio * key_dim)
        
        # softmax
        self.scale = key_dim ** -0.5
        
        # layer
        self.use_linear = layer =='linear'
        layer = LinearNorm if layer =='linear' else ConvNorm
        
        # 1. Projection 
        out_dim = (self.q_dim + self.k_dim + self.v_dim) * num_heads

        self.qkv_proj = layer(inp_dim=inp_dim, out_dim=out_dim, bias=bias)                    
        
        # 2.Merge
        merge_dim = self.v_dim * num_heads
        
        self.merge = nn.Sequential(OrderedDict(
            [   
                ('act',     act()        ),
                ('poj',     layer(inp_dim=merge_dim, out_dim=inp_dim, bias=False))
                ])
                                   )
             
    def forward(self, x):
        
        if self.use_linear:
            B, N, C = x.shape
           
            # Projection
            qkv = self.qkv_proj(x)
            
            q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.q_dim, self.k_dim, self.v_dim], dim=3)
            
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Attention 
            att = q @ k.transpose(-2, -1) * self.scale
            att = att.softmax(dim=-1)

            x = (att @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.v_dim)
            
        else:
                 
            B, C , H, W = x.shape   

            # Projection
            qkv = self.qkv_proj(x)
            
            q, k, v = qkv.view(B, self.num_heads, -1, H * W).split([self.q_dim, self.k_dim, self.v_dim], dim=2)
            
            q = q.transpose(-2, -1)  # B h N  d
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)                       
            
            # Attention 
            att = (q @ k.transpose(-2, -1)) * self.scale            # B h N N  
            att = att.softmax(dim=-1)

            x = (att @ v).transpose(-2, -1).reshape(B, -1, H, W)    # B C H W

        # Merge          
        x = self.merge(x) 
       
        return x
        
class SubSampleAttention(nn.Module):
    
    def __init__(self, inp_dim, out_dim, key_dim=None, 
                 num_heads=None, att_ratio=4, 
                 stride=1, 
                 bias=False, drop_rate=0.5, 
                 layer='conv', act=nn.GELU):
        super().__init__()
        
        self.inp_dim = inp_dim
        self.num_heads =  num_heads
        self.key_dim = key_dim
        
        # individual dims 
        self.q_dim = key_dim
        self.k_dim = key_dim
        self.v_dim = int(att_ratio * key_dim)
        
        # softmax
        self.scale = key_dim ** -0.5
        
        # layer
        self.use_linear = layer =='linear'
        layer = LinearNorm if layer =='linear' else ConvNorm
        
        # 1. Projection 
        kv_out_dim = (self.k_dim + self.v_dim) * num_heads
        q_out_dim  = self.q_dim * num_heads

        self.kv_proj = layer(inp_dim=inp_dim, out_dim=kv_out_dim, bias=bias) 
        
        self.sub_layer = Subsample(stride=stride) if self.use_linear else nn.AvgPool2d(kernel_size=1, stride=stride)
            
        self.q_proj = layer(inp_dim=inp_dim, out_dim=q_out_dim, bias=bias)
        
        # 2.Merge
        merge_dim = self.v_dim * num_heads

        self.merge = nn.Sequential(OrderedDict(
            [   
                ('act',     act()        ),
                ('poj',     layer(inp_dim=merge_dim, out_dim=out_dim, bias=False))
                ])
                                   )
                
    def forward(self, x):
        
        
        if self.use_linear: 
        
            B, N, C = x.shape
           
            # Projection
            k, v = self.kv_proj(x).view(B, N, self.num_heads, -1).split([self.k_dim, self.v_dim], dim=3)
            
            x = self.sub_layer(x)
            q = self.q_proj(x).view(B, -1, self.num_heads, self.k_dim)
            
            k = k.permute(0, 2, 1, 3)  
            v = v.permute(0, 2, 1, 3)  
            q = q.permute(0, 2, 1, 3)

            # Attention
            att = q @ k.transpose(-2, -1) * self.scale
            att = att.softmax(dim=-1)

            x = (att @ v).transpose(1, 2).reshape(B, -1, self.num_heads * self.v_dim)
            
        else:
            B, C , H, W = x.size()
            
            # Projection
            k, v = self.kv_proj(x).view(B, self.num_heads, -1, H * W).split([self.k_dim, self.v_dim], dim=2)
            
            x   = self.sub_layer(x)
            q   = self.q_proj(x).view(B,  self.num_heads, self.key_dim, -1)
            
            # Attention 
            att = (q.transpose(-2, -1) @ k) * self.scale
            att = att.softmax(dim=-1)

            # new resolution 
            h = (H + 1) // 2
            w = (W + 1) // 2

            x = (v @ att.transpose(-2, -1)).view(B, -1, h, w) 
        
        #
        x = self.merge(x)
        
        return x
    
                   
class Encoder(nn.Module):
    
    def __init__(self, inp_dim, key_dim=None, num_heads=4, att_ratio=4,
                 drop_rate=0., mlp_ratio=2, 
                 attention='full', layer="conv", act=nn.GELU):
        super().__init__()
        
        # linear or conv layers
        self.use_linear = layer=="linear"
         
        # A)  Attention 
        attention = Attention(inp_dim=inp_dim, key_dim=key_dim, num_heads=num_heads, att_ratio=att_ratio, 
                              drop_rate=drop_rate, 
                              layer=layer, act=act)
        
        # Attention Residual Block
        self.Attention_Block = ResidualBlock(attention, drop=drop_rate)
                               
        # output same
        out_dim = inp_dim
        
        # B)  MLP
        mlp_layer = MLP if layer=="linear" else  ConvMlp
        
        mlp = mlp_layer(dim=out_dim,mlp_ratio=mlp_ratio,drop_rate=drop_rate, bias=False, act=act) 

        # MLP Residual Block
        self.MLP_Block = ResidualBlock(mlp, drop=drop_rate)                                         

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        
        # Attention 
        x = self.Attention_Block(x)
        
        # MLP
        x = self.MLP_Block(x)   
        
        return x
    

class SubSample(nn.Module):
    
    def __init__(self, inp_dim, out_dim, key_dim=None, num_heads=4, att_ratio=4,
                 do_subsample=False, stride=1,  
                 drop_rate=0., mlp_ratio=2, attention='full', layer="conv", act=nn.GELU):
        super().__init__()
        
        # linear or conv layers
        self.use_linear = layer=="linear"
         
        # A)  Attention 
        attention = SubSampleAttention(inp_dim=inp_dim, out_dim=out_dim, key_dim=key_dim, num_heads=num_heads, att_ratio=att_ratio, 
                                        stride=stride, drop_rate=drop_rate, 
                                        layer=layer, act=act)
    
        
        # Attention Block
        self.Attention_Block = nn.Sequential(OrderedDict(
                              [
                                  ('attention',       attention),
                                #   ('drop',            nn.Dropout(drop_rate))
                                  ])
                                                     )

        # FF 
        mlp_layer = MLP if layer=="linear" else  ConvMlp

        mlp = mlp_layer(dim=out_dim,mlp_ratio=mlp_ratio, drop_rate=drop_rate, bias=False, act=act) 

        # MLP Residual Block
        self.MLP_Block = nn.Sequential(OrderedDict(
                        [
                            ('mlp',           mlp),
                            # ('drop',          nn.Dropout(drop_rate))
                            ])
                                               )                                      

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """            
         
        x = self.Attention_Block(x)

        x = self.MLP_Block(x)
        
        return x
