# core
from retrieval.utils.misc        import OTHER_LAYERS, NORM_LAYERS
from retrieval.utils.misc        import scheduler_from_config

from retrieval.modules.pools import RET_LAYERS
# 
import torch
import torch.optim as optim
import torch.nn as nn
from  torch.optim.lr_scheduler import ExponentialLR, LambdaLR 
from  torch.optim import Optimizer

# logger
from loguru import logger


def build_optimizer(cfg, model):
    
    optim_cfg   = cfg["optimizer"]
    
    # params groups
    params = model.parameter_groups(optim_cfg)

   
    all_params = []
    for x in params:
        all_params.extend(x['params'])
        

    assert len(all_params) == len([p for p in model.parameters() if p.requires_grad]), \
          "not all parameters that require grad are accounted for in the optimizer" 
    
    # optimizer
    if optim_cfg.get("type") == 'SGD':
        optimizer   = optim.SGD(params, nesterov=optim_cfg.getboolean("nesterov"))
    elif optim_cfg.get("type") == 'Adam':
        optimizer   = optim.Adam(params)
    elif optim_cfg.get("type") == 'AdamW':
        optimizer   = optim.AdamW(params)
    else:
        raise KeyError("unrecognized optimizer {}".format(optim_cfg["type"]))


    return optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
     
                       
class BurnInLR(_LRScheduler):
    def __init__(self, base, steps, start):
        self.base = base
        self.steps = steps
        self.start = start
        super(BurnInLR, self).__init__(base.optimizer, base.last_epoch)

    def step(self, epoch=None):
        super(BurnInLR, self).step(epoch)

        # Also update epoch for the wrapped scheduler
        if epoch is None:
            epoch = self.base.last_epoch + 1
        self.base.last_epoch = epoch

    def get_lr(self):
        beta = self.start
        alpha = (1. - beta) / self.steps
        if self.last_epoch <= self.steps:
            return [base_lr * (self.last_epoch * alpha + beta) for base_lr in self.base_lrs]
        else:
            return self.base.get_lr()
        
        
def build_lr_scheduler(cfg, optimizer):
    """
        Build a LR scheduler from config.
    """
    
    scheduler_types = ["linear", "exp"]
    
    scheduler_cfg   = cfg["scheduler"]
    
    assert scheduler_cfg["type"] in scheduler_types

    params      = scheduler_cfg.getstruct("params")
    num_epochs  = scheduler_cfg.getint("epochs")

    # linear
    if scheduler_cfg["type"] == "linear":
        beta    = float(params["from"])
        alpha   = float(params["to"] - beta) / num_epochs

        scheduler = LambdaLR(optimizer, lambda it: it * alpha + beta)
    
    # exponential
    elif scheduler_cfg["type"] == "exp":
        scheduler = ExponentialLR(optimizer,
                                gamma=params["gamma"])
    else:
        raise ValueError(f"unrecognized scheduler type {scheduler_cfg['type']}, valid options: {scheduler_types}")
    
    # warm up
    if scheduler_cfg.getint("burn_in_steps") != 0:
        scheduler = BurnInLR(scheduler,
                             scheduler_cfg.getint("burn_in_steps"),
                             scheduler_cfg.getfloat("burn_in_start"))
    print(scheduler.optimizer)
    return scheduler
