from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from retrieval.utils.burn_lr import BurnInLR

def build_lr_scheduler(cfg, optimizer):
    """ build lr scheduler """

    scheduler_types = ["linear", "exp"]
    scheduler_cfg = cfg["scheduler"]

    assert scheduler_cfg["type"] in scheduler_types

    params = scheduler_cfg.params
    num_epochs = scheduler_cfg.epochs

    if scheduler_cfg["type"] == "linear":
        beta = float(params["from"])
        alpha = float(params["to"] - beta) / num_epochs
        scheduler = LambdaLR(optimizer, lambda it: it * alpha + beta)
    
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
