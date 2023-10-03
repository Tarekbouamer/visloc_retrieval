from collections import OrderedDict

import torch
import torch.distributed as dist

from . import scheduler as lr_scheduler


def scheduler_from_config(cfg, optimizer, epoch_length):
    assert cfg.scheduler.type in (
        "linear", "step", "exp", "poly", "multistep")

    params = cfg.scheduler.params

    if cfg.scheduler.type == "linear":
        if cfg.scheduler["update_mode"] == "batch":
            count = epoch_length * cfg.scheduler.epochs
        else:
            count = cfg.scheduler.epochs

        beta = float(params["from"])
        alpha = float(params["to"] - beta) / count

        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: it * alpha + beta)

    elif cfg.scheduler.type == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        params["step_size"],
                                        params["gamma"])

    elif cfg.scheduler.type == "exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                               gamma=params["gamma"])
    elif cfg.scheduler.type == "poly":
        if cfg.scheduler["update_mode"] == "batch":
            count = epoch_length * cfg.scheduler.epochs
        else:
            count = cfg.scheduler.epochs
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: (1 - float(it) / count) ** params["gamma"])

    elif cfg.scheduler.type == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             params["milestones"],
                                             params["gamma"])

    else:
        raise ValueError(f"Unrecognized scheduler type {cfg.scheduler.type}, \
                          valid options: 'linear', 'step', 'poly', 'multistep'")

    if cfg.scheduler.burn_in_steps != 0:
        scheduler = lr_scheduler.BurnInLR(scheduler,
                                          cfg.scheduler.burn_in_steps,
                                          cfg.scheduler.burn_in_start)

    return scheduler


def freeze_params(module):
    """Freeze all parameters of the given module"""
    for p in module.parameters():
        p.requires_grad_(False)


def all_reduce_losses(losses):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in losses.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.cat([v.view(1) for v in values], dim=0)
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.view(())) for k, v in zip(names, values))


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list
