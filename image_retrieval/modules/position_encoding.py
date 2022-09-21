import math
import torch
from torch import device, nn


class PositionEncodingSine(nn.Module):
    """
        This is a sinusoidal position encoding 
    """

    def __init__(self, embedding_size, max_shape=(128, 128)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super(PositionEncodingSine, self).__init__()

        pe = torch.zeros((embedding_size, *max_shape))
        
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        
        div_term = torch.exp(torch.arange(0, embedding_size//2, 2).float() * (-math.log(10000.0) / embedding_size//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """ 

        return x + self.pe[:, :, :x.size(2), :x.size(3)].to(device=x.device)