
from loguru import logger

from retrieval.loss import ContrastiveLoss, TripletLoss

LOSSES = ["triplet", "contrastive"]


def build_loss(cfg):
    """ Build loss function """

    # Create Loss
    logger.debug(f"creating Loss function {cfg.loss.type}")

    loss_name = cfg.loss.type
    loss_margin = cfg.loss.margin

    # Triplet
    if loss_name == 'triplet':
        return TripletLoss(margin=loss_margin)
    # Constractive
    elif loss_name == 'contrastive':
        return ContrastiveLoss(margin=loss_margin)
    else:
        raise NotImplementedError(
            f"Loss {loss_name} not implemented, available: {LOSSES}")
