import torch.optim as optim


def build_optimizer(cfg, model):
    """ build optimizer """

    # params groups
    params = model.parameter_groups()

    all_params = []
    for x in params:
        all_params.extend(x['params'])

    assert len(all_params) == len([p for p in model.parameters() if p.requires_grad]), \
        "not all parameters that require grad are accounted for in the optimizer"

    # optimizer
    if cfg.optimizer.type == 'SGD':
        optimizer = optim.SGD(
            params, nesterov=cfg.optimizer.getboolean("nesterov"))
    elif cfg.optimizer.type == 'Adam':
        optimizer = optim.Adam(params)
    elif cfg.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(params)
    else:
        raise KeyError(f"unrecognized optimizer {cfg.optimizer.type}")

    return optimizer
