

from torch.utils.data import DataLoader

from retrieval.models import create_retrieval


class BaseExtractor:
    def __init__(self, cfg, model_name=None, model=None):
        self.cfg = cfg

        # model
        if model is None:
            assert model_name is not None, "model name or model must be provided"
            model = create_retrieval(model_name, pretrained=True)

        # eval mode
        self.model = model.eval().to(self.device)

    def eval(self):
        """ eval mode """
        if self.model.training:
            self.model.eval()

    def make_dataloader(self, iterable):
        """ make dataloader"""
        if isinstance(iterable, DataLoader):
            return iterable
        return DataLoader(iterable, num_workers=1, shuffle=False)

    def extract(self, **kwargs):
        raise NotImplementedError
