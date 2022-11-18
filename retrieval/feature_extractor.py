import time
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from retrieval.utils.logging import setup_logger
from retrieval.datasets import ImagesListDataset
from retrieval.models import create_model, get_pretrained_cfg


# logger
import logging
logger = logging.getLogger("retrieval")

__DEBUG__ = False

    
class FeatureExtractorOptions(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor():
    """ Feature extraction 
    """
    def __init__(self, model_name=None, model=None, cfg=None):
        super().__init__()
        
        # options
        self.options = FeatureExtractorOptions()
        
        # 
        if model is not None:
            self.model  = model
            self.cfg    = cfg
        #
        elif model_name is not None:
            self.cfg    = get_pretrained_cfg(model_name)
            self.model  = create_model(model_name, pretrained=True)
        #
        else:
            self.model  = None
            self.cfg    = None
        
        # 
        self.out_dim = self.cfg['global'].getint('out_dim')
           
        # set to device
        self.model = self.__cuda__(self.model)
        
        # set to eval mode
        self.eval()

        # transform
        self.transform = transforms.Compose([
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ])
      
    def eval(self):
        if self.model.training:
            self.model.eval()
            
    def __init_writer__(self, save_path):  
        self.writer = h5py.File(str(save_path), 'a')
      
    def __write__(self, key, data, name='desc'):   
        hf  = self.writer
        try:
            if key in hf:
                del hf[key]
                    
            g = hf.create_group(key)
            g.create_dataset(name, data=data)

        except OSError as error:   
            raise error   
    
    def __cuda__(self, x):
        if torch.cuda.is_available():
            return x.cuda()
        else:
            return x
        
    def __prepare_input__(self, x, normalize=False):
        
        # normalize
        if normalize:
            x = transforms(x)

        # BCHW
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        
        return x
    
    def __to_numpy__(self, x):
                    
        if x.is_cuda:
            x = x.cpu()
            
        return x.numpy()
    
    def __check_size__(self, x, min_size=100, max_size=2000):
        # too large (area)
        if not (x.size(-1) * x.size(-2) <= max_size * max_size):
            return True
        
        # too small
        if not (x.size(-1) >= min_size and x.size(-2) >= min_size):
            return True
        
        return False
    
    def __resize__(self, x, scale=1.0, mode='bilinear'):
        if scale == 1.0:
            return x
        else:
            return functional.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)
    
    def __dataloader__(self, dataset):
        if isinstance(dataset, DataLoader):
            return dataset
        else:
            return DataLoader(dataset, num_workers=1, shuffle=False, drop_last=False, pin_memory=True)
    
    @torch.no_grad()     
    def extract_global(self, dataset, scales=[1.0], save_path=None, normalize=False, min_size=100, max_size=2000):
        
        # to eval
        self.eval()
        
        # file writer
        if save_path is not None:
            self.writer = h5py.File(str(save_path), 'a')

        # dataloader
        dataloader = self.__dataloader__(dataset)

        # L D
        features = np.empty(shape=(len(dataloader), self.out_dim))

        # time
        start_time = time.time()
        
        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='magenta', desc='extract global'.rjust(15))):
            
            #
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img, normalize=normalize) 
            img  = self.__cuda__(img) 
            
            # D
            desc = torch.zeros(self.out_dim)
            desc = self.__cuda__(desc) 
            
            #
            num_scales = 0. 
            
            # extract --> 
            for scale in scales:
                
                # resize
                img_s = self.__resize__(img, scale=scale)
                
                # assert size within boundaries
                if self.__check_size__(img_s, min_size, max_size):
                    continue
                
                num_scales += 1.0
                 
                # extract globals
                preds   = self.model.extract_global(img_s, do_whitening=True)
                desc_s  = preds['feats'].squeeze(0)
                
                # add
                desc +=  desc_s
            
            # normalize
            desc = (1.0 / num_scales) * desc
            desc = functional.normalize(desc, dim=-1)
            
            # numpy
            features[it]    = self.__to_numpy__(desc) 
            
            # write
            if hasattr(self, 'writer'):
                name = data['img_name'][0]
                self.__write__(name, desc)
            
            # clear cache  
            if it % 10 == 0:
                torch.cuda.empty_cache()
                
        # close writer    
        if hasattr(self, 'writer'):  
            self.writer.close()  
        
        # end time
        end_time = time.time() - start_time  
        
        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')
        
        #
        out = {
            'features':     features,
            'save_path':    save_path
            }
        
        return out
    
    @torch.no_grad()     
    def extract_locals(self, dataset, num_features=50, scales=[1.0], save_path=None, normalize=False, min_size=100, max_size=2000):

        # to eval
        self.eval()
        
        # file writer
        if save_path is not None:
            self.writer = h5py.File(str(save_path), 'a')
        
        # dataloader
        dataloader = self.__dataloader__(dataset)
        
        # L N D
        features = np.empty(shape=(len(dataloader), num_features, self.out_dim))
        imids = []

        # time
        start_time = time.time()
        
        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='green', desc='extract locals'.rjust(15))):
            
            #
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img) 
            img  = self.__cuda__(img) 
            
            # N D
            desc = torch.zeros(num_features, self.out_dim)
            desc = self.__cuda__(desc) 

            # counter 
            num_scales = 0. 
            
            # scale -->
            for scale in scales:
                
                # resize
                img_s = self.__resize__(img, scale=scale)
                
                # assert size to boundaries
                if self.__check_size__(img_s, min_size, max_size):
                    continue
                
                num_scales += 1.0
                 
                # extract locals 
                preds   = self.model.extract_locals(img_s, num_features=num_features)
                desc_s  = preds['feats']

                # add
                desc += desc_s
            
            # normalize
            desc = (1.0 / num_scales) * desc
            desc = functional.normalize(desc, dim=-1)

            # numpy
            features[it]    = self.__to_numpy__(desc)
            imids.append(np.full((desc.shape[0],), it))
            
            # write
            if hasattr(self, 'writer'):
                name = data['img_name'][0]
                self.__write__(name, desc)
            
            # clear cache  
            if it % 10 == 0:
                torch.cuda.empty_cache()
        
        # close writer    
        if hasattr(self, 'writer'):  
            self.writer.close()  
        
        # end time
        end_time = time.time() - start_time  
        
        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')
        
        #
        features    = features.reshape((-1, self.out_dim))
        ids         = np.hstack(imids)
        
        #
        out = {
            'features':     features,
            'ids':  ids,
            'save_path':    save_path
            }
        
        return out
                                    
          
            
if __name__ == '__main__':
    
    
    logger = setup_logger(output=".", name="retrieval")
    # DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
    DATA_DIR='/media/loc/ssd_5127/tmp/how/how_data/test/oxford5k/jpg'
    save_path='db.h5'
    
    dataset = ImagesListDataset(DATA_DIR, max_size=500)
    
    # feature_extractor = FeatureExtractor("resnet50_c4_gem_1024")
    feature_extractor = FeatureExtractor("resnet50_c4_how")

    scales = [0.707, 1.0, 1.424]
    print(feature_extractor.model)
    
    global_feat, pth = feature_extractor.extract_global(dataset,    scales=scales, save_path=save_path)
    local_feat, pth = feature_extractor.extract_locals(dataset,     scales=scales, save_path=save_path)
    
    print("features path", pth)
    print("global features shape", global_feat.shape)
    print("global features shape", local_feat.shape)

        
  