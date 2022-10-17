
import torch.nn as nn


class Head(nn.Module):
    
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

    def __init__(self, inp_dim, out_dim):
        super(Head, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
                     
    def reset_parameters(self):
            
        for name, mod in self.named_modules():
                
            if isinstance(mod,  nn.Linear):
                nn.init.xavier_normal_(mod.weight, 1.0)
                    
            elif isinstance(mod, nn.LayerNorm):
                nn.init.constant_(mod.weight, 0.01)

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
                
        raise NotImplementedError