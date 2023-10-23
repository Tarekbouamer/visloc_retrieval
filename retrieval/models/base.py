import torch
import torch.nn as nn
from omegaconf import OmegaConf


class RetrievalBase(nn.Module):
    """Base class for retrieval models"""
    
    # model required inputs
    required_inputs = ["image"]

    # model default config
    default_cfg = {}

    # paper reference
    paper_ref = ""

    # code reference
    code_ref = ""

    def __init__(self, cfg):
        super(RetrievalBase, self).__init__()

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)

    @property
    def out_dim(self):
        """Return output dimension"""
        return self._out_dim

    @property
    def feature_dim(self):
        """Return feature dimension"""
        return self._feature_dim
    
    @property
    def device(self):
        """Return torch device"""
        return next(self.parameters()).device

    @torch.no_grad()
    def init_model(self, **kwargs):
        """Initialize model"""
        pass

    def parameter_groups(self, **kwargs):
        """Return torch parameter groups"""
        params = [{
            'params':          [p for p in self.parameters() if p.requires_grad],
            'lr':              self.cfg.optimizer.lr,
            'weight_decay':    self.cfg.optimizer.weight_decay}]

        return params

    def transform_inputs(self, data: dict) -> dict:
        """Transform inputs"""
        raise NotImplementedError

    def forward(self,  data=None, **kwargs):
        """"""
        raise NotImplementedError

    @torch.no_grad()
    def extract(self, data=None, **kwargs):
        """"""
        raise NotImplementedError
