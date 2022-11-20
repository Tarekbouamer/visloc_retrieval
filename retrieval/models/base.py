from typing import List
import torch.nn as nn

class BaseNet(nn.Module):
    """ ImageRetrievalNet

        General image retrieval model, consists of backbone and head
    
    """
    def __init__(self, body, head, init_model=None):
        super(BaseNet, self).__init__()
        
        self.body   = body
        self.head   = head
        
        #
        self._init_model = init_model
    
    def device(self):
        return next(self.parameters()).device
            
    def forward(self, img=None, do_whitening=True):
        raise NotImplementedError

    def extract_global(self, img=None, do_whitening=True):
        raise NotImplementedError