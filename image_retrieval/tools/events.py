

import logging

from collections import defaultdict, OrderedDict

import torch
import tensorboardX as tensorboard
from torchvision.utils import make_grid
# logger
import logging
logger = logging.getLogger("retrieval")

from math import log10     
 
def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


class Meter:
    def __init__(self):
        self._states = OrderedDict()
        

    def register_state(self, name, tensor):
        if name not in self._states and isinstance(tensor, torch.Tensor):
            self._states[name] = tensor

    def __getattr__(self, item):
        if "_states" in self.__dict__:
            _states = self.__dict__["_states"]
            if item in _states:
                return _states[item]

    def reset(self):
        for state in self._states.values():
            state.zero_()

    def state_dict(self):
        return dict(self._states)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self._states:
                self._states[k].copy_(v)
            else:
                raise KeyError("Unexpected key {} in state dict when loading {} from state dict"
                               .format(k, self.__class__.__name__))


class ConstantMeter(Meter):
    def __init__(self, shape):
        super(ConstantMeter, self).__init__()
        self.register_state("last", torch.zeros(shape, dtype=torch.float32))

    def update(self, value):
        self.last.copy_(value)

    @property
    def value(self):
        return self.last


class AverageMeter(ConstantMeter):
    def __init__(self, shape=(), momentum=1.):
        super(AverageMeter, self).__init__(shape)
        self.register_state("sum",      torch.zeros(shape,  dtype=torch.float32))
        self.register_state("count",    torch.tensor(0,     dtype=torch.float32))
        
        self.momentum = momentum

    def update(self, value):
        super(AverageMeter, self).update(value)
        self.sum.mul_(self.momentum).add_(value)
        self.count.mul_(self.momentum).add_(1.)
  
    @property
    def mean(self):
        if self.count.item() == 0:
            return torch.tensor(0.)
        else:
            return self.sum / self.count.clamp(min=1)
        
        
class Writer:
    """
        Base class for writers that obtain events from :class:`EventStorage` and process them.
    """
    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class EventWriter(Writer):
    """
    """

    def __init__(self, directory, print_freq=10):

        
        self.summray    = tensorboard.SummaryWriter(directory)
        self.logger     = logging.getLogger(__name__)
        
        # 
        self.print_freq = print_freq
        self.histories  = OrderedDict()
        
        # 
        for mode in ["train", "eval", "test" ]:
            self.histories[mode] = defaultdict(AverageMeter)       
        
        # 
        self.is_training = True
        
    def train(self):
        self.mode = "train"
        self.is_training = True
        
        self.history = self.histories["train"]

    def eval(self):
        self.mode = "eval"
        self.is_training = False
        
        self.history = self.histories["eval"]

    def test(self):
        self.mode = "test"
        self.is_training = False
        
        self.history = self.histories["test"]

    def add_scalar(self, name, value, iter):
        # summary board
        self.summray.add_scalar(name, value, iter)
  
    def add_scalars(self, data, iter):
        for k , v in data.items():
            self.add_scalar(self.mode + "/" + k, v, iter)

    def add_images(self, images, iter):
        for id , im in enumerate(images):
            self.summray.add_images(f'tuple/{id}', im, iter)

    def add_graph(self, model, images=None):
        images = images[0].cuda()
        self.summray.add_graph(model, images)
        
    def put(self, name, value):
        # update history
        history = self.history[name]
        history.update(value)
    
    def get(self):
        return dict(self.history)
    
    def set(self, state_dicts):
        # for name, meter in meters.items():
        #     meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        NotImplementedError
        
    def write(self, global_step):
        for k, v in self.history.items():
            if isinstance(v, AverageMeter):
                self.summray.add_scalar(self.mode + "/" + k, v.value.item(), global_step)
         
    def log(self, epoch, num_epochs, step, num_steps):
        """
            Print metrics and stats to terminal
        """
        # epoch and steps
        msg  = _current_total_formatter(epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)

        # metrics
        for k, v in self.history.items():
            if isinstance(v, AverageMeter):
                msg += "\t{}={:.3f} ({:.3f})".format(k, v.value.item(), v.mean.item())

        # memory usage megabites
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0        
            msg  += "\t mem={:.2f}".format(max_mem_mb)

        # log
        logger.info(msg)        
            


