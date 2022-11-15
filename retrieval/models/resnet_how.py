from .registry  import register_model, get_pretrained_cfg
from .factory   import create_model, load_pretrained

from typing import List
from copy import deepcopy

import os
import torch 
import torch.nn.functional as functional
from tqdm import tqdm
import numpy as np

import timm
from timm.utils.model               import freeze, unfreeze
from retrieval.modules.heads        import create_head
from retrieval.models.base          import BaseNet

from retrieval.datasets     import  INPUTS 
from retrieval.utils.pca    import PCA

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
        pred    = model.extract_locals(**batch, do_whitening=False)

        vecs.append(pred['feats'].cpu().numpy())
            
        del pred
    
    # stack
    vecs  = np.vstack(vecs)
        
    logger.info('compute PCA')
    
    #
    m, P  = PCA(vecs)
    m, P = m.T, P.T
        
    # create layer
    layer   = deepcopy(model_head.whiten)  
    print(layer)
    num_d   = layer.weight.shape[0]
    
    #
    projection          = torch.Tensor(P[: num_d, :]).unsqueeze(-1).unsqueeze(-1)
    projected_shift     = - torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze() 

    #
    layer.weight.data   = projection
    layer.bias.data     = projected_shift[:num_d]

    # save layer to layer_path
    layer_path = os.path.join(args.directory, "whiten.pth")
    torch.save(layer.state_dict(), layer_path)

    logger.info(f"save whiten layer: {layer_path}")
            
    # load   
    model.head.whiten.load_state_dict(layer.state_dict())
     
    #        
    logger.info("pca done")

 
def _create_model(variant, body_name, head_name, cfg=None, pretrained=True, feature_scales=[1, 2, 3, 4], **kwargs):
    
    # assert
    assert body_name in timm.list_models(pretrained=True), f"model: {body_name}  not implemented timm models yet!"

    # get flags
    reduction   = cfg["global"].getint("reduction")
    frozen      = cfg.pop("frozen", [])
    
    # body
    body = timm.create_model(body_name, 
                             features_only=True, 
                             out_indices=feature_scales, 
                             pretrained=True) # always pretrained
    
    #
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
    if reduction > 0: 
        assert reduction < out_dim, (f"reduction {reduction} has to be less than input dim {out_dim}")
        out_dim = reduction
    
    # head
    head = create_head(head_name, 
                        inp_dim=body_dim,
                        out_dim=out_dim,
                        **kwargs)
    
    # model
    model = HowNet(body, head, init_model=_init_model) 
    
    # 
    if pretrained:
        pretrained_cfg = get_pretrained_cfg(variant)
        load_pretrained(model, variant, pretrained_cfg) 
        
    logger.info(f"body channels:{body_channels}  reductions:{body_reductions}   layer_names: {body_module_names}")
    
    #
    if cfg:
        cfg.set('global', 'global_dim', str(out_dim))
 
    return model

# 
class HowNet(BaseNet):
    
    def how_select_local(self, ms_feats, ms_masks, *, scales, num_feats):
        """
            Convert multi-scale feature maps with attentions to a list of local descriptors
                :param list ms_feats: A list of feature maps, each at a different scale
                :param list ms_masks: A list of attentions, each at a different scale
                :param list scales: A list of scales (floats)
                :param int features_num: Number of features to be returned (sorted by attenions)
                :return tuple: A list of descriptors, attentions, locations (x_coor, y_coor) and scales where
                        elements from each list correspond to each other
        """
        device = ms_feats[0].device
        
        size = sum(x.shape[0] * x.shape[1] for x in ms_masks)

        desc = torch.zeros(size, ms_feats[0].shape[1], dtype=torch.float32, device=device)
        atts = torch.zeros(size, dtype=torch.float32, device=device)
        locs = torch.zeros(size, 2, dtype=torch.int16, device=device)
        scls = torch.zeros(size, dtype=torch.float16, device=device)

        pointer = 0
        for sc, vs, ms in zip(scales, ms_feats, ms_masks):
            
            #
            if len(ms.shape) == 0:
                continue
            
            #
            height, width = ms.shape
            numel = torch.numel(ms)
            slc = slice(pointer, pointer+numel)
            pointer += numel

            #
            desc[slc] = vs.squeeze(0).reshape(vs.shape[1], -1).T
            atts[slc] = ms.reshape(-1)
            width_arr = torch.arange(width, dtype=torch.int16)
            locs[slc, 0] = width_arr.repeat(height).to(device) # x axis
            height_arr = torch.arange(height, dtype=torch.int16)
            locs[slc, 1] = height_arr.view(-1, 1).repeat(1, width).reshape(-1).to(device) # y axis
            scls[slc] = sc

        #
        keep_n = min(num_feats, atts.shape[0]) if num_feats is not None else atts.shape[0]
        idx = atts.sort(descending=True)[1][:keep_n]

        #
        preds = {
            'feats':    desc[idx],
            'attns':    atts[idx],
            'locs':     locs[idx],
            'cls':      scls[idx]
            }
        
        #
        return preds
    
    def l2n(self, x, eps=1e-6):
      return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)
    
    def weighted_spoc(self, ms_feats, ms_weights):
        """
            Weighted SPoC pooling, summed over scales.
                :param list ms_feats: A list of feature maps, each at a different scale
                :param list ms_weights: A list of weights, each at a different scale
                :return torch.Tensor: L2-normalized global descriptor
        """
        
        desc = torch.zeros((1, ms_feats[0].shape[1]), dtype=torch.float32, device=ms_feats[0].device)
        
        for feats, weights in zip(ms_feats, ms_weights):
            desc += (feats * weights).sum((-2, -1)).squeeze()
        
        desc = self.l2n(desc)
        
        preds = {
            'feats' : desc
        }
        
        return preds

    def __forward__(self, img, scales=[1], do_whitening=True):
        
        #
        feats_list, attns_list = [], []
        
        #
        for s in scales:
            #
            imgs = functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
            
            #
            x = self.body(imgs)
            
            #
            if isinstance(x, List):
                x = x[-1] 
            
            # head 
            preds = self.head(x, do_whitening=do_whitening)
            
            # 
            feats_list.append(preds['feats'])
            attns_list.append(preds['attns'])
         
        # normalize to max weight
        mx          = max(x.max()   for x in attns_list)
        attns_list   = [x/mx         for x in attns_list]
        
        #
        return feats_list, attns_list
   
    def extract_global(self, img, scales=[1], do_whitening=True):
        
        # 
        feat_list, attns_list = self.__forward__(img, scales=scales, do_whitening=do_whitening)
        
        #    
        return self.weighted_spoc(feat_list, attns_list)
    
    def extract_locals(self, img, scales=[1], num_feats=1000, do_whitening=True):
        
        #        
        feat_list, attns_list = self.__forward__(img, scales=scales, do_whitening=do_whitening)
        
        #
        return self.how_select_local(feat_list, attns_list, scales=scales, num_feats=num_feats)
    
    def forward(self, img, do_whitening=True):
        return self.extract_global(img, do_whitening=do_whitening)
            
    
# SfM-120k
@register_model
def resnet18_c4_how(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-18 with GeM model, only 4 features scales
    """   
    model_args = dict(**kwargs)
    return _create_model('resnet18_c4_how', 'resnet18', 'how', cfg, pretrained, feature_scales, **model_args)


@register_model
def resnet50_c4_how(cfg=None, pretrained=True, feature_scales=[1, 2, 3], **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model, only 4 features scales
    """   
    model_args = dict(**kwargs)
    return _create_model('resnet50_c4_how', 'resnet50', 'how', cfg, pretrained, feature_scales, **model_args)


if __name__ == '__main__':    
    create_fn = create_model("resnet50_c4_how", pretrained=True)
    print(create_fn)
