from core.device import get_device
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from retrieval.models import create_retrieval


class BaseExtractor:
    # device
    device = get_device()

    def __init__(self, cfg, model_name=None, model=None):

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)
        
        # model
        if model is None:
            self.model = create_retrieval(cfg, model_name)
        else:
            self.model = model

        # eval mode
        self.model.eval()

        # to device
        self.model = self.model.to(self.device)

    def eval(self):
        """ eval mode """
        if self.model.training:
            self.model.eval()

    def make_dataloader(self, iterable):
        """ make dataloader"""
        if isinstance(iterable, DataLoader):
            return iterable
        return DataLoader(iterable, num_workers=8, shuffle=False)

    def extract(self, **kwargs):
        raise NotImplementedError
