import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """ Triplet Loss
    """
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        
        self.margin = margin

    def forward(self, x, target):

        # x is N x D
        dim = x.size(1)
                
        # Number of tuples            
        nq = torch.sum(target.data==-1).item() 
        
        # Number of images per tuple Nq + Np + Nn
        S = x.size(0) // nq 

        #
        xa = x[target.data==-1, :].repeat(1,S-2).view((S-2)*nq,  dim)
        xp = x[target.data==1,  :].repeat(1,S-2).view((S-2)*nq,  dim)
        xn = x[target.data==0,  :]

        #
        dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=1)
        dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=1)

        #
        return torch.sum(torch.clamp(dist_pos - dist_neg + self.margin, min=0))

    def __repr__(self):
        return f'{self.__class__.__name__}(margin={self.margin:.4})'