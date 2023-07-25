
from retrieval.loss import TripletLoss, ContrastiveLoss

# logger
from loguru import logger

def build_loss(cfg):
    
    # parse params with default values
    global_config = cfg["global"]

    # Create Loss
    logger.debug("creating Loss function { %s }", global_config.get("loss"))
    
    loss_name       = global_config.get("loss")
    loss_margin     = global_config.getfloat("loss_margin")
    
    # Triplet
    if loss_name == 'triplet':
        return TripletLoss(margin=loss_margin)
    
    # Constractive   
    elif loss_name == 'contrastive':
        return ContrastiveLoss(margin=loss_margin)
    
    else:
        raise NotImplementedError(f"loss not implemented yet {global_config.get('loss') }" )