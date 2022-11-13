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


# TODO: remove it
def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def weighted_spoc(ms_feats, ms_weights):
    """
        Weighted SPoC pooling, summed over scales.
            :param list ms_feats: A list of feature maps, each at a different scale
            :param list ms_weights: A list of weights, each at a different scale
            :return torch.Tensor: L2-normalized global descriptor
    """
    
    desc = torch.zeros((1, ms_feats[0].shape[1]), dtype=torch.float32, device=ms_feats[0].device)
    
    for feats, weights in zip(ms_feats, ms_weights):
        desc += (feats * weights).sum((-2, -1)).squeeze()
    
    return l2n(desc)


def how_select_local(ms_feats, ms_masks, *, scales, features_num):
    """
        Convert multi-scale feature maps with attentions to a list of local descriptors
            :param list ms_feats: A list of feature maps, each at a different scale
            :param list ms_masks: A list of attentions, each at a different scale
            :param list scales: A list of scales (floats)
            :param int features_num: Number of features to be returned (sorted by attenions)
            :return tuple: A list of descriptors, attentions, locations (x_coor, y_coor) and scales where
                    elements from each list correspond to each other
    """
    device = ms_feats[0].device
    size = sum(x.shape[0] * x.shape[1] for x in ms_masks)

    desc = torch.zeros(size, ms_feats[0].shape[1], dtype=torch.float32, device=device)
    atts = torch.zeros(size, dtype=torch.float32, device=device)
    locs = torch.zeros(size, 2, dtype=torch.int16, device=device)
    scls = torch.zeros(size, dtype=torch.float16, device=device)

    pointer = 0
    for sc, vs, ms in zip(scales, ms_feats, ms_masks):
        if len(ms.shape) == 0:
            continue

        height, width = ms.shape
        numel = torch.numel(ms)
        slc = slice(pointer, pointer+numel)
        pointer += numel

        desc[slc] = vs.squeeze(0).reshape(vs.shape[1], -1).T
        atts[slc] = ms.reshape(-1)
        width_arr = torch.arange(width, dtype=torch.int16)
        locs[slc, 0] = width_arr.repeat(height).to(device) # x axis
        height_arr = torch.arange(height, dtype=torch.int16)
        locs[slc, 1] = height_arr.view(-1, 1).repeat(1, width).reshape(-1).to(device) # y axis
        scls[slc] = sc

    keep_n = min(features_num, atts.shape[0]) if features_num is not None else atts.shape[0]
    idx = atts.sort(descending=True)[1][:keep_n]

    return desc[idx], atts[idx], locs[idx], scls[idx]


# 
class L2Attention(nn.Module):
    """ 
        attention as L2-norm of local descriptors
    """

    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)

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
        attns = self.attention(x)
        attns = nn.functional.normalize(attns, dim=-1, p=2, eps=1e-6)

        # pool
        x = self.pool(x)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # withen 
        if do_whitening:
            x = self.whiten(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # out
        out = {
            'feats':    x,
            'attns':    attns
            }
        
        return out      


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