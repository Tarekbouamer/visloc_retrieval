import torch


def save_snapshot(save_path,
                  epoch: int,
                  last_score: float,
                  best_score: float,
                  global_step: int,
                  config: dict = None,
                  **kwargs) -> bool:
    """Save a snapshot of the current training state

        Args:
            save_path (str): path to save the snapshot
            epoch (int): current epoch
            last_score (float): last score
            best_score (float): best score
            global_step (int): current global step
            config (dict, optional): config dictionary. Defaults to None.

        Returns:
            bool: True if the snapshot is saved successfully
    """
    # create data dictionary
    data = {
        "config": config,
        "epoch": epoch,
        "last_score": last_score,
        "best_score": best_score,
        "global_step": global_step,
        **dict(kwargs)}
    
    try:
        # save snapshot
        torch.save(data, save_path)
    except Exception:
        return False

    return True


def resume_from_snapshot(model, snapshot, modules):

    # load snapshot
    snapshot = torch.load(snapshot, map_location="cpu")

    # state_dict
    state_dict = snapshot["state_dict"]

    # load state_dict
    for module in modules:
        if module in state_dict:
            _load_pretraining_dict(getattr(model, module), state_dict[module])
        else:
            raise KeyError(
                "The given snapshot does not contain a state_dict for module '{module}'")

    return snapshot


def _load_pretraining_dict(model, state_dict):
    """Load state dictionary from a pre-training snapshot
    """
    model_sd = model.state_dict()

    for k, v in model_sd.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                del state_dict[k]

    model.load_state_dict(state_dict, False)

# def pre_train_from_snapshots(model, snapshots, modules):
#     for snapshot in snapshots:
#         if ":" in snapshot:
#             module_name, snapshot = snapshot.split(":")
#         else:
#             module_name = None

#         snapshot = torch.load(snapshot, map_location="cpu")
#         state_dict = snapshot["state_dict"]

#         if module_name is None:
#             for module_name in modules:
#                 if module_name in state_dict:
#                     _load_pretraining_dict(
#                         getattr(model, module_name), state_dict[module_name])
#         else:
#             if module_name in modules:
#                 _load_pretraining_dict(
#                     getattr(model, module_name), state_dict[module_name])
#             else:
#                 raise ValueError(
#                     f"Unrecognized network module {module_name}")
