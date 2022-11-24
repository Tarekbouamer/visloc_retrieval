import torch.nn as nn


class BaseHead(nn.Module):
    """ Global Image Descriptor Head 

        inp_dim:    int         the input features size  
        out_dim:    int         the final descriptor size
        
    """

    def __init__(self, inp_dim, out_dim):
        super(BaseHead, self).__init__()

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
            x:              torch.Tensor    input tensor [B, C, H, W] 
            do_whitening:   boolean         do or skip whithening 
        """
                
        raise NotImplementedError