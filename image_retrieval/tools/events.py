
import datetime
import json
import logging
import os
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from typing import Optional
import torch

from typing import List, Optional, Tuple

import numpy as np

import tensorboardX as tensorboard

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
    Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.
    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]
        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, directory, print_freq=10):
        
        flags = os.O_RDWR | os.O_CREAT
        mode  = 0o666
        
        self.json_file  = os.open(os.path.join(directory, "metrics"), flags, mode)
        self.summray    = tensorboard.SummaryWriter(directory)
        
        self.logger     = logging.getLogger(__name__)
        
        # 
        self.print_freq = print_freq
        self.history    = defaultdict(AverageMeter)

    def add_scalar(self, name, value, iter):
        # summary board
        self.summray.add_scalar(name, value, iter)
  
    def add_scalars(self, data, iter):
        for k , v in data.items():
            self.add_scalar(k, v, iter)

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
                self.summray.add_scalar(k, v.value.item(), global_step)
        
    def write_to_file(self, data, iter):
        
        if (iter % self.print_freq) !=0 or iter != 0:
            return
        
        for itr, scalars_per_iter in data.items():
            scalars_per_iter["iteration"] = itr
            self.json_file.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self.json_file.flush()
        
        try:
            os.fsync(self.json_file.fileno())
        except AttributeError:
            pass

    def close(self):
        self.json_file.close()
    
        
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
            


