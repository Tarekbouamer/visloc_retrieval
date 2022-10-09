from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

# core
from image_retrieval.utils.misc        import OTHER_LAYERS, NORM_LAYERS
from image_retrieval.utils.misc        import scheduler_from_config

from image_retrieval.modules.pools import RET_LAYERS
# 
import torch
import torch.optim as optim
import torch.nn as nn
from  torch.optim.lr_scheduler import ExponentialLR, LambdaLR 
from  torch.optim import Optimizer

# logger
import logging
logger = logging.getLogger("retrieval")


# def make_optimizer(model, config, epoch_length):
  
#     body_config = config["body"]

#     optim_cfg = config["optimizer"]
#     sched_cfg = config["scheduler"]

#     # Base learning rate and weight decay
#     LR              = optim_cfg.getfloat("lr")
#     WEIGHT_DECAY    = optim_cfg.getfloat("weight_decay")

#     # Tunning learning rate and weight decay
#     lr_coefs            = optim_cfg.getstruct("lr_coefs")
#     weight_decay_coefs  = optim_cfg.getstruct("weight_decay_coefs")

#     # Separate classifier parameters from all others
#     norm_params      = []
#     other_params     = []
       
#     # body
#     for k, m in model.body.named_modules():
#         if any(isinstance(m, layer) for layer in NORM_LAYERS):
#             norm_params  += [p for p in m.parameters() if p.requires_grad]
            
#         elif any(isinstance(m, layer) for layer in OTHER_LAYERS):
#             other_params += [p for p in m.parameters() if p.requires_grad]
    

#     # ret head
#     pool_params     = []
#     whiten_params   = []
        
#     for k, v in model.ret_head.named_children():
            
#         if k.find("pool") != -1:
#             pool_params += [p for p in v.parameters() if p.requires_grad]
        
#         elif k.find("whiten")!= -1:
#             whiten_params += [p for p in v.parameters() if p.requires_grad]

#     assert len(norm_params) + len(other_params) + len(pool_params) + len(whiten_params) \
#                             == len([p for p in model.parameters() if p.requires_grad]), \
#         "Not all parameters that require grad are accounted for in the optimizer"
        
#     # Set-up optimizer hyper-parameters
#     parameters = [
#         # body norm 
#         {
#             "params": norm_params,
#             "lr":           LR              if not body_config.getboolean("bn_frozen") else 0.,
#             "weight_decay": WEIGHT_DECAY    if optim_cfg.getboolean("weight_decay_norm") else 0
#         },
#         # body other 
#         {
#             "params": other_params,
#             "lr":           LR        ,
#             "weight_decay": WEIGHT_DECAY 
#         }]
    
#     # Pool
#     if len(pool_params) > 0  and lr_coefs["pool"]:
#         parameters.append(
#         {
#             "params": pool_params,
#             "lr":           LR * lr_coefs["pool"],
#             "weight_decay": WEIGHT_DECAY * weight_decay_coefs["pool"]
#         })
#     else:
#         raise TypeError(" pool parameters not optimized ")
    
#     # Whiten
#     if len(whiten_params) > 0 and lr_coefs["whiten"]:
#         parameters.append(
#         {
#             "params": whiten_params,
#             "lr":           LR * lr_coefs["whiten"],
#             "weight_decay": WEIGHT_DECAY * weight_decay_coefs["whiten"]
#         })
        
   
#     # Select optimizer
#     if optim_cfg.get("type") == 'SGD':
#         optimizer = optim.SGD(parameters,
#                               lr=LR,
#                               weight_decay=optim_cfg.getfloat("weight_decay"),
#                               nesterov=optim_cfg.getboolean("nesterov"))
#     elif optim_cfg.get("type") == 'Adam':
#         optimizer = optim.Adam(parameters,
#                                lr=LR, 
#                                weight_decay=WEIGHT_DECAY)
#     elif optim_cfg.get("type") == 'AdamW':
#         optimizer = optim.AdamW(parameters,
#                                lr=LR, 
#                                weight_decay=WEIGHT_DECAY)
#     else:
#         raise KeyError("unrecognized optimizer {}".format(optim_cfg["type"]))
#     print(optimizer)

#     # Set scheduler
#     scheduler = scheduler_from_config(sched_cfg, optimizer, epoch_length)
        
#     assert sched_cfg.get("update_mode") in ("batch", "epoch")
#     batch_update = sched_cfg.get("update_mode") == "batch"
#     total_epochs = sched_cfg.getint("epochs")

#     return optimizer, scheduler, parameters, batch_update, total_epochs


def build_optimizer(cfg, model):
    
    body_config = cfg["body"]
    optim_cfg = cfg["optimizer"]

    # Base learning rate and weight decay
    LR              = optim_cfg.getfloat("lr")
    WEIGHT_DECAY    = optim_cfg.getfloat("weight_decay")

    # body parameters
    body_norm_params      = []
    body_other_params     = []
    
    for _, m_layer in model.body.named_modules():        
    
        if any(isinstance(m_layer, layer) for layer in NORM_LAYERS):
            body_norm_params  += [p for p in m_layer.parameters() if p.requires_grad]
                
        elif any(isinstance(m_layer, layer) for layer in OTHER_LAYERS + RET_LAYERS ):
            body_other_params += [p for p in m_layer.parameters() if p.requires_grad]

    # head
    pool_params     = []
    whiten_params   = []
        
    for k, v in model.head.named_children():          
        if k.find("pool") != -1:
            pool_params += [p for p in v.parameters() if p.requires_grad]
        elif k.find("whiten")!= -1:
            whiten_params += [p for p in v.parameters() if p.requires_grad] 
    assert len(body_norm_params) + len(body_other_params) + len(pool_params) + len(whiten_params)== \
        len([p for p in model.parameters() if p.requires_grad]), \
          "not all parameters that require grad are accounted for in the optimizer"
    
    # hyper-parameters
    params = [
        # norm 
        {
            "params":       body_norm_params,
            "lr":           LR              if not body_config.getboolean("bn_frozen")  else 0.,
            "weight_decay": WEIGHT_DECAY    if optim_cfg.getboolean("weight_decay_norm") else 0
        },
        # other 
        {
            "params":       body_other_params,
            "lr":           LR        ,
            "weight_decay": WEIGHT_DECAY 
        },
        {
            "params":       pool_params,
            "lr":           LR * 10,
            "weight_decay": 0.
        },
        {
            "params":       whiten_params,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY
        }
        ]
    
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
    
    scheduler_cfg   = cfg["scheduler"]
    assert scheduler_cfg["type"] in ("linear", "exp")

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
        raise ValueError("unrecognized scheduler type {}, valid options: 'linear', 'exp', "
                         .format(scheduler_cfg["type"]))
    
    # warm up
    if scheduler_cfg.getint("burn_in_steps") != 0:
        scheduler = BurnInLR(scheduler,
                             scheduler_cfg.getint("burn_in_steps"),
                             scheduler_cfg.getfloat("burn_in_start"))

    return scheduler
