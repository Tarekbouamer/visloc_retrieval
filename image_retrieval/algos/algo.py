import numpy as np
import torch

import torch.nn as nn

from image_retrieval.modules.losses import TripletLoss, ContrastiveLoss, APLoss

from core.utils.misc import Empty
from core.utils.parallel import PackedSequence


class globalLoss:
    """
        Image Retrieval loss
    """
    def __init__(self, name=None, sigma=0.1, epsilon=1e-6):
        self.name = name
        self.sigma = sigma
        self.epsilon = epsilon
        
        if self.name == 'triplet':
            self.criterion = TripletLoss(margin=self.sigma)
        
        elif self.name == 'contrastive':
            self.criterion = ContrastiveLoss(margin=self.sigma)

        elif self.name == 'ap':
            self.criterion = APLoss()        
        else:
            NameError("Loss function type not found !")

    def __call__(self, x, target):
        
        loss = self.criterion(x, target)        
        
        return loss


class globalFeatureAlgo:
    """Base class for Image Retrieval algorithms

    """
    def __init__(self, loss, batch_size):

        self.loss = loss
        self.batch_size = batch_size

    def _get_level(self, x):
        if isinstance(x, dict):
            last_mod = next(reversed(x))  # last element mod5 or mod4
            x = x[last_mod] 
            
        else:
            x = x

        return x

    def training(self, head=None, x=None, do_whitening=True, do_local=False):

        x = self._get_level(x)
        
        try:
            if do_local:
                ret_pred = head.forward_locals(x, do_whitening)
            else:
                ret_pred = head(x, do_whitening)

        except Empty:
            ret_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return ret_pred

    def inference(self, head=None, x=None, do_whitening=True, do_local=False):
        
        x = self._get_level(x)

        try:
            if do_local:
                ret_pred = head.forward_locals(x, do_whitening)
            else:
                ret_pred = head(x, do_whitening)
        
        except Empty:
            ret_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return ret_pred