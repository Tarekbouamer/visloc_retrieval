from .registry  import register_model, get_pretrained_cfg
from .factory   import create_model, load_pretrained

from copy import deepcopy
from typing import List

import os
import torch 
import torch.nn as nn
import torch.nn.functional as functional

from tqdm import tqdm
import numpy as np

import timm
from timm.utils.model               import freeze, unfreeze

from retrieval.modules.heads        import create_head
from retrieval.models.base          import BaseNet
from retrieval.datasets             import  INPUTS 

from retrieval.utils.pca   import PCA


# logger
from loguru import logger


# default cfg          
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
    #resnet50
    'sfm_resnet50_gem_2048':        
        _cfg(drive='https://drive.google.com/uc?id=1gFRNJPILkInkuCZiCHqjQH_Xa2CUiAb5', out_dim=2048),
    
    'sfm_resnet50_c4_gem_1024':     
        _cfg(drive='https://drive.google.com/uc?id=1ber3PbTF4ZWAmnBuJu5AEp2myVJFNM7F'),
    
    # resnet101
    'sfm_resnet101_gem_2048':       
        _cfg(drive='https://drive.google.com/uc?id=10CqmzZE_XwRCyoiYlZh03tfYk1jzeziz', out_dim=2048),
   
    'sfm_resnet101_c4_gem_1024':       
        _cfg(drive='https://drive.google.com/uc?id=1uYYuLqqE9TNgtmQtY7Mg2YEIF9VkqAYz'),
       
        
    # gl18
    'gl18_resnet50_gem_2048':      
        _cfg(drive='https://drive.google.com/uc?id=1AaS4aXe2FYyi-iiLetF4JAo0iRqKHQ2Z', out_dim=2048),

    # gl20
    'gl20_resnet50_gem_2048':      
        _cfg(out_dim=2048),    
    
    'sfm_ig_resnext101_32x8d_gem':  _cfg(out_dim=960),

    # mobile 
    'sfm_mobilenetv3_large_100_gem':  _cfg(out_dim=960),
    
    #
    'sfm_tf_efficientnet_l2_ns_gem':  _cfg(out_dim=1024),

    #
    'sfm_regnetz_d8_gem':  _cfg(out_dim=2048),
    #
    'sfm_tf_efficientnet_b7_ns_gem':  _cfg(out_dim=640),
    'sfm_tf_efficientnet_b7_gem':   _cfg(out_dim=640),
    
    #
    'sfm_tf_efficientnet_b6_ns_gem':  _cfg(out_dim=576),

    #
    'sfm_tf_efficientnet_b5_gem':   _cfg(out_dim=512),
    
    #
    'sfm_regnety_160_gem':   _cfg(out_dim=2048),

    #
    'sfm_regnety_120_gem':   _cfg(out_dim=2048),

    'sfm_resnet200d_gem':   _cfg(out_dim=2048),
  
    }


# init model function
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
                
        vecs.append(pred['features'].cpu().numpy())
            
        del pred
    
    # stack
    vecs  = np.vstack(vecs)

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
    
    
# create model function   
def _create_model(variant, body_name, head_name, cfg=None, pretrained=True, feature_scales=[1, 2, 3, 4], **kwargs):
        
    # assert
    assert body_name in timm.list_models(pretrained=True), f"model: {body_name}  not implemented timm models yet!"

    # default cfg
    default_cfg = get_pretrained_cfg(variant)
    out_dim     = default_cfg.pop('out_dim', None)
    frozen      = default_cfg.pop("frozen", [])

    # body
    body = timm.create_model(body_name, 
                             features_only=True, 
                             out_indices=feature_scales, 
                             pretrained=True) # always pretrained
 
    
    body_channels           = body.feature_info.channels()
    body_reductions         = body.feature_info.reduction()
    body_module_names       = body.feature_info.module_name()
    
    # output dim
    body_dim = body_channels[-1]
   
   # freeze layers
    if len(frozen) > 0:
        frozen_layers = [ body_module_names[item] for item in frozen]
        logger.info(f"frozen layers {frozen_layers}")
        freeze(body, frozen_layers) 
    
    # reduction 
    assert out_dim <= body_dim, (f"reduction {out_dim} has to be less than or equal to input dim {body_dim}")
    
    # head
    head = create_head(head_name, 
                        inp_dim=body_dim,
                        out_dim=out_dim,
                        **kwargs)
    
    # model
    model = GemNet(body, head, init_model=_init_model) 
    
    # 
    if pretrained:
        load_pretrained(model, variant, default_cfg) 
        
    logger.info(f"body channels:{body_channels}  reductions:{body_reductions}   layer_names: {body_module_names}")
    
    #
    if cfg:
        cfg.set('global', 'global_dim', str(out_dim))
 
    return model


# GemNet 
class GemNet(BaseNet):
    """ Gem Model
    """
    
    def parameter_groups(self, optim_cfg):
        """
            Return torch parameter groups
        """
        
        # base 
        LR              = optim_cfg.getfloat("lr")
        WEIGHT_DECAY    = optim_cfg.getfloat("weight_decay")

        # base layer
        layers = [self.body, self.head.whiten]
        
        # base params 
        params = [{
                    'params':          [p for p in x.parameters() if p.requires_grad],
                    'lr':              LR,
                    'weight_decay':    WEIGHT_DECAY }   for x in layers]
        
        # 10x faster, no regularization
        if self.head.pool:
            params.append({
                            'params':       [p for p in self.head.pool.parameters() if p.requires_grad], 
                            'lr':           10*LR,
                            'weight_decay': 0.}
                          )
            
        return params
            
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
    
    def __sum_scales(self, features):
        """ sum over scales """
        #
        desc = torch.zeros((1, features[0].shape[1]), dtype=torch.float32, device=features[0].device)
        
        # 
        for vec in features:
            desc += vec 

        # normalize
        desc = functional.normalize(desc, dim=-1)
        
        preds = {
            'features': desc
        }
        return preds

    def __forward__(self, img, scales=[1], do_whitening=True):
        
        # 
        features = []

        # --> 
        for scale in scales:
            
            # resize
            img_s = self.__resize__(img, scale=scale)

            # assert size within boundaries
            if self.__check_size__(img_s):
                continue
            
            # -->
            preds = self.forward(img_s, do_whitening)
            
            # 
            features.append(preds['features'])
        
        # sum over scales
        return self.__sum_scales(features)
        
    def extract_global(self, img, scales=[1], do_whitening=True):
        return self.__forward__(img, scales=scales, do_whitening=do_whitening)
        
    
# SfM-120k
@register_model
def sfm_mobilenetv3_large_100_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k mobilenetv3_large_100 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_mobilenetv3_large_100_gem', 'mobilenetv3_large_100', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_tf_efficientnet_l2_ns_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k mobilenetv3_large_100 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_tf_efficientnet_l2_ns_gem', 'tf_efficientnet_l2_ns', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_regnetz_d8_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k regnetz_d8	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_regnetz_d8_gem', 'regnetz_d8', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_resnet200d_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k resnet200d	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet200d_gem', 'resnet200d', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_tf_efficientnet_b7_ns_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k tf_efficientnet_b7_ns	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_tf_efficientnet_b7_ns_gem', 'tf_efficientnet_b7_ns', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_tf_efficientnet_b7_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k tf_efficientnet_b7	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_tf_efficientnet_b7_gem', 'tf_efficientnet_b7', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_tf_efficientnet_b5_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k tf_efficientnet_b5	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_tf_efficientnet_b5_gem', 'tf_efficientnet_b5', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_tf_efficientnet_b6_ns_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k tf_efficientnet_b6_ns	with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_tf_efficientnet_b6_ns_gem', 'tf_efficientnet_b6_ns', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_regnety_160_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k regnety_160 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_regnety_160_gem', 'regnety_160', 'gem' , cfg, pretrained, **model_args)


@register_model
def sfm_regnety_120_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k regnety_120 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_regnety_120_gem', 'regnety_120', 'gem' , cfg, pretrained, **model_args)




@register_model
def sfm_ig_resnext101_32x8d_gem(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_ig_resnext101_32x8d_gem', 'ig_resnext101_32x8d', 'gem', cfg, pretrained, **model_args)


@register_model
def sfm_resnet18_gem_512(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet18_gem_512', 'resnet18', 'gem', cfg, pretrained, **model_args)


@register_model
def sfm_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model.
    """  
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_gem_2048', 'resnet50', 'gem', cfg, pretrained, **model_args)


@register_model
def sfm_resnet50_c4_gem_1024(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model, only 4 features scales
    """   
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_c4_gem_1024', 'resnet50', 'gem', cfg, pretrained, feature_scales, **model_args)


@register_model
def sfm_resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model.
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_gem_2048', 'resnet101', 'gem', cfg, pretrained, **model_args)


@register_model
def sfm_resnet101_c4_gem_1024(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model, only 4 features scales
    """    
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_c4_gem_1024', 'resnet101', 'gem', cfg, pretrained, feature_scales, **model_args)

#Google Landmark 18
@register_model
def gl18_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a gl18 ResNet-50 with GeM model.
    """  
    model_args = dict(**kwargs)
    return _create_model('gl18_resnet50_gem_2048', 'resnet50', 'gem', cfg, pretrained, **model_args)

# Google Landmark 20
@register_model
def gl20_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a gl20 ResNet-50 with GeM model.
    """  
    model_args = dict(**kwargs)
    return _create_model('gl20_resnet50_gem_2048', 'resnet50', 'gem', cfg, pretrained, **model_args)

if __name__ == '__main__':    
    create_fn = create_model("resnet50_c4_gem_1024", pretrained=True, p=1.234)
    print(create_fn)
