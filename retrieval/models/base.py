import torch.nn as nn
import torch.nn.functional as functional

class BaseNet(nn.Module):
    """ BaseNet

        General image retrieval model, consists of backbone and head
    
    """
    def __init__(self, body, head, init_model=None):
        super(BaseNet, self).__init__()
        
        self.body   = body
        self.head   = head
        
        # initializing the model function , mainly the head
        self._init_model = init_model

    def __check_size__(self, x, **kwargs):
        
        #
        min_size = kwargs.pop('min_size', 0)
        max_size = kwargs.pop('max_size', 2000)

        # too large (area)
        if not (x.size(-1) * x.size(-2) <= max_size * max_size):
            return True
        
        # too small
        if not (x.size(-1) >= min_size and x.size(-2) >= min_size):
            return True
        
        return False
    
    def __resize__(self, x, scale=1.0, mode='bilinear'):
        if scale == 1.0:
            return x
        else:
            return functional.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)
        
    def device(self):
        return next(self.parameters()).device
    
    def parameter_groups(self, optim_cfg):
        """Return torch parameter groups"""
        raise NotImplementedError

    def forward(self, img=None, do_whitening=True):
        raise NotImplementedError

    def extract_global(self, img=None, do_whitening=True):
        raise NotImplementedError
    
    def extract_locals(self, img, num_features=1000, do_whitening=True):
        raise NotImplementedError