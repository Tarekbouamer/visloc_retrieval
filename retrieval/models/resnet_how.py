from .registry  import register_model, get_pretrained_cfg
from .factory   import create_model, load_pretrained

import timm
from timm.utils.model               import freeze, unfreeze
from retrieval.modules.heads        import create_head
from retrieval.models.base          import ImageRetrievalNet


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
    }

 
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
    model = ImageRetrievalNet(body, head) 
    
    # 
    if pretrained:
        pretrained_cfg = get_pretrained_cfg(variant)
        load_pretrained(model, variant, pretrained_cfg) 
        
    logger.info(f"body channels:{body_channels}  reductions:{body_reductions}   layer_names: {body_module_names}")
    
    #
    if cfg:
        cfg.set('global', 'global_dim', str(out_dim))
 
    return model
    

# SfM-120k
@register_model
def resnet50_c4_how(cfg=None, pretrained=True, **kwargs):
    """
        Constructs a SfM-120k ResNet-50 with How head.
    """  
    model_args = dict(**kwargs)
    return _create_model('resnet50_c4_how', 'resnet50', 'how', cfg, pretrained, **model_args)


if __name__ == '__main__':    
    create_fn = create_model("resnet50_c4_how", pretrained=True)
    print(create_fn)
