from copy import deepcopy
import imghdr
from typing import List
import torch
import torch.nn as nn

from  tqdm import tqdm
 
import numpy as np
from collections import OrderedDict

from image_retrieval.datasets.generic import ImagesFromList, ImagesTransform, INPUTS
from core.utils.parallel import PackedSequence
from core.utils.PCA         import PCA, PCA_whitenlearn_shrinkage


class _ImageRetrievalNet(nn.Module):
    
    def __init__(self, body, ret_head, num_features=100):
        super(ImageRetrievalNet, self).__init__()
        
        self.body = body

        self.ret_head = ret_head
        
        self.num_features = num_features

    def _prepare_sequence(self, q, p, ns):
        
        inp_data    = PackedSequence(q) 
        inp_data    += PackedSequence(p)
        
        for n in ns:
            inp_data +=  PackedSequence(n)

        return inp_data

    def _prepare_tuple(self, q, p, ns):
        inp_data = []
        
        inp_data.extend([q])
        inp_data.extend([p])
        inp_data.extend(ns)
        
        return inp_data
    
    def init_whitening(self, state, logger=None):
        logger.debug("Init whitening layer")
        self.ret_head.whiten.load_state_dict(state)
         
    def compute_whitening(self, dl, layer_name, device, logger=None):
        
        logger("Compute Whitening {%s}", layer_name) 
        self.eval()
        self.to(device=device)
        
        if  layer_name not in ["local", "global"]:
            raise ValueError(" layer should be local or global ")
        #
        inp_dim = self.ret_head.inp_dim
        out_dim = self.ret_head.global_dim if layer_name == "global" else self.ret_head.local_dim
        
        # 
        do_local = layer_name == "local"
        
        with torch.no_grad():
            
            # Prepare query loader
            logger('Extracting descriptors for PCA {%s}--{%s} for {%s}:', inp_dim, out_dim, len(dl))

            # Extract  vectors
            vecs = []
            for it, batch in tqdm(enumerate(dl), total=len(dl)):
  
                # Upload batch
                batch   = {k: batch[k].cuda(device=device, non_blocking=True) for k in INPUTS}
                pred    = self(**batch, do_whitening=False, do_local=do_local)
                
                vecs.append(pred["descs"].cpu().numpy())
                del pred
            #
            vecs  = np.vstack(vecs)
      
        logger('Compute PCA, Takes a while')
    
        m, P  = PCA_whitenlearn_shrinkage(vecs)
        m, P = m.T, P.T
        
        # create layer
        if layer_name == "global":
            layer = deepcopy(self.ret_head.whiten)  
        else :
            layer = deepcopy(self.ret_head.local_whiten)
        
        projection          = torch.Tensor(P[:layer.weight.shape[0], :])
        projected_shift     = - torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        
        if layer_name == "local":
            projection = projection.unsqueeze(-1).unsqueeze(-1)
            
        layer.weight.data   = projection.to(layer.weight.device)
        layer.bias.data     = projected_shift[:layer.weight.shape[0]].to(layer.bias.device)    

        return layer
           
    def extract_body_features(self, img, scales):
        
        if not isinstance(scales, List):
            raise TypeError('scales should be list of scale factors, example [0.5, 1.0, 2.0]')        
        
        #
        features, sizes = [], []
        for s in scales :
            
            s_img = nn.functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
            
            # body
            x = self.body(s_img)
            
            # features level, last
            if isinstance(x, List):
                x = x[-1] 
            
            # append           
            features.append(x)
            sizes.append(s_img.shape[-2:])
        
        return features, sizes

    def random_sampler(self, L):
    
        K           = min(self.num_features, L) if self.num_features is not None else L
        indices     = torch.randperm(L)[:K]  
        
        return indices  
                
    def extract_local_features(self, features, scales, do_whitening):
        """
            Extract local descriptors at each scale 
        """
        
        # list of decs 
        descs_list, locs_list = self.ret_head.forward_locals(features, scales, do_whitening)
        
        # ---> tensor
        descs = torch.cat(descs_list,   dim=1).squeeze(0)
        locs  = torch.cat(locs_list,    dim=1).squeeze(0)
        
        # sample 

        L   = descs.shape[0]
        idx = self.random_sampler(L)
                
        return descs[idx], locs[idx]
      
    def extract_global_features(self, features, scales, do_whitening):
        
        descs_list = self.ret_head.forward(features, scales,  do_whitening)

        # ---> tensor
        descs = torch.cat(descs_list,   dim=0)
                
        # sum and normalize
        descs   = torch.sum(descs, dim=0).unsqueeze(0)
        descs   = nn.functional.normalize(descs, dim=-1, p=2, eps=1e-6)
        
        return descs
        
    def forward(self, img=None, scales=[1.0],  do_whitening=False, do_local=False, **varargs):


        # Run network body
        features, img_sizes = self.extract_body_features(img, scales=scales)

        # Run head
        descs, locs =None, None
        
        if do_local:
            descs, locs = self.extract_local_features(features, scales, do_whitening)
        else:
            descs = self.extract_global_features(features, scales, do_whitening)
            
        pred = OrderedDict([
            ("descs", descs),
            ("locs", locs)
        ])

        return pred
    

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, body, head):
        super(ImageRetrievalNet, self).__init__()
        
        self.body   = body
        self.head   = head
        
    def forward(self, img=None, do_whitening=True):
          
        # body
        x = self.body(img)
        
        if isinstance(x, List):
            x = x[-1] 
        
        # head
        preds = self.head(x, do_whitening)
        
        if self.training:
            return preds

        return preds
        
