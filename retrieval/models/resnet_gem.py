from .registry  import register_model, get_pretrained_cfg
from .factory   import create_model, load_pretrained

from copy import deepcopy
from typing import List

import os
import torch 
import torch.nn as nn

from tqdm import tqdm
import numpy as np

import timm
from timm.utils.model               import freeze, unfreeze

from retrieval.modules.heads        import create_head
from retrieval.models.base          import BaseNet
from retrieval.datasets             import  INPUTS 

from retrieval.utils.pca   import PCA


# logger
import logging
logger = logging.getLogger("retrieval")



def _cfg(url='', drive='', out_dim=1024, **kwargs):
    return {
        'url': url,
        'drive':drive,
        'reduction': False, 
        'input_size': (3, 1024, 1024),
        'out_dim': out_dim, 
        **kwargs
    }
 
 
default_cfgs = {
    #sfm resnet50
    'resnet50_gem_2048':        
        _cfg(drive='https://drive.google.com/uc?id=1gFRNJPILkInkuCZiCHqjQH_Xa2CUiAb5', out_dim=2048),
    'resnet50_c4_gem_1024':     
        _cfg(drive='https://drive.google.com/uc?id=1ber3PbTF4ZWAmnBuJu5AEp2myVJFNM7F'),
    
    # sfm resnet101
    'resnet101_gem_2048':       
        _cfg(drive='https://drive.google.com/uc?id=10CqmzZE_XwRCyoiYlZh03tfYk1jzeziz', out_dim=2048),
   
    'resnet101_c4_gem_1024':       
        _cfg(drive='https://drive.google.com/uc?id=1uYYuLqqE9TNgtmQtY7Mg2YEIF9VkqAYz'),
       
        
    # gl18
    'gl18_resnet50_gem_2048':      
        _cfg(drive='https://drive.google.com/uc?id=1AaS4aXe2FYyi-iiLetF4JAo0iRqKHQ2Z', out_dim=2048)
    }


@torch.no_grad()
def _init_model(args, cfg, model, sample_dl):
    
    # eval   
    if model.training:
        model.eval()   
    
    # options 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 
    model_head  = model.head
    inp_dim     = model_head.inp_dim
    out_dim     = model_head.out_dim

            
    # prepare query loader
    logger.info(f'extracting descriptors for PCA {inp_dim}--{out_dim} for {len(sample_dl)}')

    # extract vectors
    vecs = []
    for _, batch in tqdm(enumerate(sample_dl), total=len(sample_dl)):
  
        # upload batch
        batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
        pred    = model(**batch, do_whitening=False)
                
        vecs.append(pred['feats'].cpu().numpy())
            
        del pred
    
    # stack
    vecs  = np.vstack(vecs)
    print(vecs.shape)
    
    logger.info('compute PCA')
    
    m, P  = PCA(vecs)
    m, P = m.T, P.T
        
    # create layer
    layer = deepcopy(model_head.whiten)  
    data_size = layer.weight.data.size()
    
    #
    num_d  = layer.weight.shape[0]
    
    #
    projection      = torch.Tensor(P[: num_d, :]).view(data_size)
    projected_shift = - torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze() 

    #
    layer.weight.data   = projection
    layer.bias.data     = projected_shift[:num_d]

    # save layer to whithen_path
    layer_path = os.path.join(args.directory, "whiten.pth")
    torch.save(layer.state_dict(), layer_path)
    
    logger.info(f"save whiten layer: {layer_path}")
            
    # load   
    model.head.whiten.load_state_dict(layer.state_dict())
            
    logger.info("pca done")
    
    return 
    
 
def _create_model(variant, body_name, head_name, cfg=None, pretrained=True, feature_scales=[1, 2, 3, 4], **kwargs):
    
    # assert
    assert body_name in timm.list_models(pretrained=True), f"model: {body_name}  not implemented timm models yet!"

    # get flags
    reduction   = kwargs.pop("reduction", False)
    frozen      = kwargs.pop("frozen", [])
    
    # body
    body = timm.create_model(body_name, 
                             features_only=True, 
                             out_indices=feature_scales, 
                             pretrained=True) # always pretrained
    
    body_channels           = body.feature_info.channels()
    body_reductions         = body.feature_info.reduction()
    body_module_names       = body.feature_info.module_name()
    
    # output dim
    body_dim = out_dim = body_channels[-1]
   
   # freeze layers
    if len(frozen) > 0:
        frozen_layers = [ body_module_names[item] for item in frozen]
        logger.info(f"frozen layers {frozen_layers}")
        freeze(body, frozen_layers) 
    
    # reduction 
    if reduction:
        assert reduction < out_dim, (f"reduction {reduction} has to be less than input dim {out_dim}")
        out_dim = reduction
    
    # head
    head = create_head(head_name, 
                        inp_dim=body_dim,
                        out_dim=out_dim,
                        **kwargs)
    
    # model
    model = GemNet(body, head, init_model=_init_model) 
    
    # 
    if pretrained:
        pretrained_cfg = get_pretrained_cfg(variant)
        load_pretrained(model, variant, pretrained_cfg) 
        
    logger.info(f"body channels:{body_channels}  reductions:{body_reductions}   layer_names: {body_module_names}")
    
    #
    if cfg:
        cfg.set('global', 'global_dim', str(out_dim))
 
    return model

##
class GemNet(BaseNet):
    """ ImageRetrievalNet

        General image retrieval model, consists of backbone and head
    
    """
        
    def forward(self, img=None, do_whitening=True):
          
        # body
        x = self.body(img)
        
        #
        if isinstance(x, List):
            x = x[-1] 
        
        # head
        preds = self.head(x, do_whitening)
        
        if self.training:
            return preds

        return preds
    
    
# SfM-120k
@register_model
def resnet10t_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-10-T with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('resnet10t_gem', 'resnet10t', 'gem_linear' , cfg, pretrained, **model_args)


@register_model
def resnet18_gem_512(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('resnet18_gem_512', 'resnet18', 'gem_linear', cfg, pretrained, **model_args)


@register_model
def resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model.
    """  
    model_args = dict(**kwargs)
    return _create_model('resnet50_gem_2048', 'resnet50', 'gem_linear', cfg, pretrained, **model_args)


@register_model
def resnet50_c4_gem_1024(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model, only 4 features scales
    """   
    model_args = dict(**kwargs)
    return _create_model('resnet50_c4_gem_1024', 'resnet50', 'gem_linear', cfg, pretrained, feature_scales, **model_args)


@register_model
def resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('resnet101_gem_2048', 'resnet101', 'gem_linear', cfg, pretrained, **model_args)


@register_model
def resnet101_c4_gem_1024(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model, only 4 features scales
    """    
    model_args = dict(**kwargs)
    return _create_model('resnet101_c4_gem_1024', 'resnet101', 'gem_linear', cfg, pretrained, feature_scales, **model_args)

# TODO: Google Landmark 18
@register_model
def gl18_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a gl18 ResNet-50 with GeM model.
    """  
    model_args = dict(**kwargs)
    return _create_model('gl18_resnet50_gem_2048', 'resnet50', 'gem_linear', cfg, pretrained, **model_args)

# TODO: Google Landmark 20


if __name__ == '__main__':    
    create_fn = create_model("resnet50_c4_gem_1024", pretrained=True, p=1.234)
    print(create_fn)
