from typing import List
import torch.nn as nn

class ImageRetrievalNet(nn.Module):
    """ ImageRetrievalNet

        General image retrieval model that consists of backbone and head
    
    """
    
    def __init__(self, body, head):
        super(ImageRetrievalNet, self).__init__()
        
        self.body   = body
        self.head   = head
        
    def forward(self, img=None, do_whitening=True):
          
        # body
        x = self.body(img)
        
        if isinstance(x, List):
            x = x[-1] 
        
        # head
        preds = self.head(x, do_whitening)
        
        if self.training:
            return preds

        return preds