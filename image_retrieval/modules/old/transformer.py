import copy
from typing_extensions import OrderedDict
import torch
import torch.nn as nn

from image_retrieval.modules.attention import LinearAttention, FullAttention


class Transformer(nn.Module):
    """
         Transformer .
    """

    def __init__(self, embedding_size, num_head, layer_names, attention ):
        super().__init__()

        self.embedding_size     = embedding_size
        self.num_head           = num_head
        self.layer_names        = layer_names
        self.attention          = attention
        
        # Encoder
        Encoder = Encoder(self.embedding_size , 
                                     self.num_head, 
                                     self.attention)
        
        # Layers
        self.layers = nn.ModuleList([copy.deepcopy(Encoder) for _ in range(len(self.layer_names))])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.embedding_size == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
