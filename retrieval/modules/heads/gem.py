
import torch.nn as nn

from retrieval.modules.pools import GeM

from .base import BaseHead
from .factory import create_head
from .registry import register_head


class GemHead(BaseHead):
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

    def __init__(self, inp_dim, out_dim, **kwargs):
        super(GemHead, self).__init__(inp_dim, out_dim)

        #
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        #
        p = kwargs.pop("p", 3.0)

        # pooling
        self.pool = GeM(p=p)

        # whitening
        self.whiten = nn.Linear(inp_dim, out_dim, bias=True)

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

        #
        preds = {
            'features': x
        }

        return preds


@register_head
def gem(inp_dim, out_dim, **kwargs):
    return GemHead(inp_dim, out_dim, layer="linear", **kwargs)


# test
if __name__ == '__main__':

    create_fn = create_head("gem_conv", 100, 10, p=1.22)
    print(create_fn)
