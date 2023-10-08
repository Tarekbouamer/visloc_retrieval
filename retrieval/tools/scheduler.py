#
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{i}] when resuming an optimizer")
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
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
    """ Burn in learning rate """

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
    """ build lr scheduler """

    scheduler_types = ["linear", "exp"]
    scheduler_cfg = cfg["scheduler"]

    assert scheduler_cfg["type"] in scheduler_types

    params = scheduler_cfg.params
    num_epochs = scheduler_cfg.epochs

    # linear
    if scheduler_cfg["type"] == "linear":
        beta = float(params["from"])
        alpha = float(params["to"] - beta) / num_epochs

        scheduler = LambdaLR(optimizer, lambda it: it * alpha + beta)

    # exponential
    elif scheduler_cfg["type"] == "exp":
        scheduler = ExponentialLR(optimizer,
                                  gamma=params["gamma"])
    else:
        raise ValueError(
            f"unrecognized scheduler type {scheduler_cfg['type']}, valid options: {scheduler_types}")

    # warm up
    if scheduler_cfg.burn_in_steps != 0:
        scheduler = BurnInLR(scheduler,
                             scheduler_cfg.burn_in_steps,
                             scheduler_cfg.burn_in_start)
    return scheduler
