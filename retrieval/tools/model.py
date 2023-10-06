from os import makedirs, path

import torch
from loguru import logger

from retrieval.utils.io import create_withen_file_from_cfg





def compute_pca(args, cfg, model, sample_dl,  **varargs):

    # Path
    whithen_folder = path.join(args.directory, "whithen")

    if not path.exists(whithen_folder):
        logger.info("Save whithening layer: %s ", whithen_folder)
        makedirs(whithen_folder)

    # Whithen_path
    whithen_path = create_withen_file_from_cfg(cfg, whithen_folder, logger)

    # Avoid recomputing same layer for further experiments
    if (not path.isfile(whithen_path) or cfg.pca.update):

        # Compute layer
        whiten_layer = run_pca(
            model, sample_dl, device=varargs["device"], logger=logger)

        # Save layer to whithen_path
        logger.info("Save whiten layer: %s ", whithen_path)

        torch.save(whiten_layer.state_dict(), whithen_path)

    # load from whithen_path
    logger.info("Load whiten state: %s ", whithen_path)
    layer_state = torch.load(whithen_path, map_location="cpu")

    # Init model layer
    model.init_whitening(layer_state, logger)
