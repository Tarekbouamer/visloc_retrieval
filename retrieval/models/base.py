import torch.nn as nn
from omegaconf import OmegaConf


class RetrievalBase(nn.Module):
    """Base class for retrieval models"""

    def __init__(self, cfg):
        super(RetrievalBase, self).__init__()

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)

    def __check_size__(self, x, min_size=0, max_size=2000):
        """Check if the size of x is too small or too large"""

        # too large (area)
        if not (x.size(-1) * x.size(-2) <= max_size * max_size):
            return True

        # too small (area)
        if not (x.size(-1) >= min_size and x.size(-2) >= min_size):
            return True

        return False

    def device(self):
        return next(self.parameters()).device

    def parameter_groups(self, **kwargs):
        """Return torch parameter groups"""
        raise NotImplementedError

    def transform_inputs(self, data: dict) -> dict:
        raise NotImplementedError

    def forward(self,  data=None, **kwargs):
        raise NotImplementedError

    def extract(self, data=None, **kwargs):
        raise NotImplementedError
