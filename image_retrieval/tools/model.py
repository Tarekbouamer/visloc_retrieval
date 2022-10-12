from os import makedirs, path
from tqdm import tqdm
import shutil
from copy import deepcopy

import numpy as np

import  torch 
import  torch.nn as nn 



# Timm 
import timm
from timm.utils.model import freeze, unfreeze

# image retrieval
from image_retrieval.datasets.tuples import ImagesFromList, ImagesTransform, INPUTS
from image_retrieval.modules.heads.head         import RetrievalHead


from image_retrieval.models.GF_net              import ImageRetrievalNet
from image_retrieval.models.GF_net              import ImageRetrievalNet

from image_retrieval.utils.io   import create_withen_file_from_cfg
from image_retrieval.utils.pca   import PCA_whitenlearn_shrinkage

# logger
import logging
logger = logging.getLogger("retrieval")

def run_pca(arg, cfg, model, sample_dl):
    
    if model.training:
        model.eval()   
    
    # options 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #
    model_head  = model.head
    inp_dim     = model_head.inp_dim
    out_dim     = model_head.out_dim

    with torch.no_grad():
            
        # prepare query loader
        logger.info(f'extracting descriptors for PCA {inp_dim}--{out_dim} for {len(sample_dl)}')

        # extract vectors
        vecs = []
        for _, batch in tqdm(enumerate(sample_dl), total=len(sample_dl)):
  
            # upload batch
            batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
            pred    = model(**batch, do_whitening=False)
                
            vecs.append(pred.cpu().numpy())
            
            del pred
    #
    vecs  = np.vstack(vecs)
    
    logger.info('compute PCA, ')
    
    m, P  = PCA_whitenlearn_shrinkage(vecs)
    m, P = m.T, P.T
        
    # create layer
    whithen_layer = deepcopy(model_head.whiten)  
    num_d  = whithen_layer.weight.shape[0]
    
    projection          = torch.Tensor(P[: num_d, :])
    projected_shift     = - torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
    
    whithen_layer.weight.data   = projection.to(device)
    whithen_layer.bias.data     = projected_shift[:num_d].to(device)    

    return whithen_layer
    
     
def compute_pca(args, cfg, model, sample_dl):
    
    # 
    body_cfg        = cfg["body"]
    global_cfg      = cfg["global"]
    data_cfg        = cfg["dataloader"]

    # Path    
    whithen_folder = path.join(args.directory, "whithen")

    if not path.exists(whithen_folder):
        logger.info("Save whithening layer: %s ", whithen_folder)
        makedirs(whithen_folder)
    
    # Whithen_path
    whithen_path = create_withen_file_from_cfg(cfg, whithen_folder, logger)

    # Avoid recomputing same layer for further experiments
    if ( not path.isfile(whithen_path) or global_cfg.getboolean("update") ):
                     
        # Compute layer
        whiten_layer = run_pca(model, sample_dl, device=varargs["device"], logger=logger)
        
        # Save layer to whithen_path
        logger.info("Save whiten layer: %s ", whithen_path)

        torch.save(whiten_layer.state_dict(), whithen_path)

    # load from whithen_path
    logger.info("Load whiten state: %s ", whithen_path)
    layer_state = torch.load(whithen_path, map_location="cpu")
    
    # Init model layer    
    model.init_whitening(layer_state, logger)
                
 
def build_model(cfg):
    
    # parse params with default values
    global_cfg   = cfg["global"]
    
    # create backbone
    body_arch       = cfg["body"].get("arch")
    feature_scales  = cfg["body"].getstruct("features_scales")
    
    logger.info(f"creating backbone model:  {body_arch}     features_scales:    {feature_scales}    Pre-trained:    {str(cfg['body'].getboolean('pretrained'))}")
    
    # check if in timm
    timm_model_list = timm.list_models(pretrained=True)
    
    # assert
    assert body_arch in timm_model_list, (f"model: {body_arch}  not implemented timm models yet!")

    # load model state dictionary
    body = timm.create_model(body_arch, 
                             features_only=True, 
                             out_indices=feature_scales, 
                             pretrained=cfg["body"].getboolean("pretrained"))
    
    body_channels           = body.feature_info.channels()
    body_reductions         = body.feature_info.reduction()
    body_module_names       = body.feature_info.module_name()
      
    logger.info("body channels: %s    reductions: %s      layer_names: %s",    body_channels, 
                                                                                body_reductions,
                                                                                body_module_names)
    # freeze modules in backbone
    if len(cfg["body"].getstruct("num_frozen")) > 0:
        logger.info("frozen layers: %s ", cfg["body"].getstruct("num_frozen"))
        frozen_layers = cfg["body"].getstruct("num_frozen")
        freeze(body, frozen_layers)
    
    # output dim
    body_dim    = body_channels[-1]
    out_dim     = body_channels[-1]
    
    # reduction 
    if global_cfg.getboolean("reduction"):
        out_dim = global_cfg.getint("global_dim")

    # head
    global_head = RetrievalHead(inp_dim=body_dim,
                                out_dim=out_dim,
                                pooling=global_cfg.getstruct("pooling"),
                                layer=global_cfg.get("type"))
    
    # create a generic image retrieval network
    model = ImageRetrievalNet(body, global_head)  
    
    # 
    cfg.set('global', 'global_dim', str(out_dim))
    
    return model

# def build_model(cfg):
    
#     create_fn = model_entrypoint("resnet50_c4_gem")

    