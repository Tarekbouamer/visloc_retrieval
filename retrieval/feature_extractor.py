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
            self.model      = model
            self.cfg        = cfg
            self.out_dim    = self.cfg['global'].getint('out_dim')
        #
        elif model_name is not None:
            self.cfg        = get_pretrained_cfg(model_name)
            self.model      = create_model(model_name, pretrained=True)
            self.out_dim    = self.cfg.pop('out_dim', 0)
        #
        else:
            self.model   = None
            self.cfg     = None
            self.out_dim = 0
          
        # set to device
        self.model = self.__cuda__(self.model)

        # set to eval mode
        self.eval()

        # transform, if needed, by default No
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
        else: # cpu
            return x
        
    def __prepare_input__(self, x, **kwargs):
        
        #
        normalize = kwargs.pop('normalize', False)
        
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
    
    def __dataloader__(self, x, **kwargs):
        if isinstance(x, DataLoader):
            return x
        else:
            return DataLoader(x, num_workers=1, shuffle=False, drop_last=False, pin_memory=True)
    
    def load_global(self, save_path):
        
        fd      = h5py.File(save_path, 'r')
        names   = list(fd.keys())
        
        descs = []
        for it, name in enumerate(tqdm(names, total=len(names), colour='magenta', desc='load global'.rjust(15))):
            descs.append(fd[name]['desc'].__array__())
         
        #   
        features = torch.from_numpy(np.stack(descs, 0)).float()
        
        #
        out = {
            'features': features,
            'save_path': save_path,
        }
        
        return out 
    
    @torch.no_grad()     
    def extract_global(self, dataset, scales=[1.0], save_path=None, **kwargs):
        """
            global features extractor
            
            dataset:        ImageFromList or data.Dataloader    list of images 
            scales:         List        extraction scales
            save_path:      str         path to write and save features hdf5 format
            normalize:      boolean     to normalize input data if not normalized 
            min_size:       int         min size of images, skip if less than min_size
            max_size:       int         max area of images, skip if more than max_size square 

        """
        
        # to eval
        self.eval()
        
        if save_path is not None:
            # load, no extaction 
            if kwargs.pop('override', False):
                return self.load_global(save_path)
            else:
                # writer
                self.writer = h5py.File(str(save_path), 'a')

        # dataloader
        dataloader = self.__dataloader__(dataset, **kwargs)

        # L D size
        features = []

        # time
        start_time = time.time()
        
        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='magenta', desc='extract global'.rjust(15))):
            
            #
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img, **kwargs) 
            img  = self.__cuda__(img) 
            
            # extract 
            desc = self.model.extract_global(img, scales=scales, do_whitening=True, **kwargs)
            desc = desc['features']
            
            # numpy
            desc = self.__to_numpy__(desc)
            features.append(desc)
            
            # write
            if hasattr(self, 'writer'):
                name = data['name'][0]
                self.__write__(name, desc)
            
            # clear cache  
            if it % 10 == 0:
                torch.cuda.empty_cache()
                
        # close writer    
        if hasattr(self, 'writer'):  
            self.writer.close()  
        
        # stack         
        features    = np.vstack(features)
        
        # end time
        end_time = time.time() - start_time  
        
        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')
        
        # 
        out = {
            'features':     features,
            'save_path':    save_path   }
        
        return out
    
    @torch.no_grad()     
    def extract_locals(self, dataset, num_features=1000, scales=[1.0], save_path=None, **kwargs):
        """
            local features extractor
            
            dataset:        ImageFromList or data.Dataloader    list of images 
            scales:         List        extraction scales
            num_features:   int         max number of local features
            save_path:      str         path to write and save features hdf5 format
            normalize:      boolean     to normalize input data if not normalized 
            min_size:       int         min size of images, skip if less than min_size
            max_size:       int         max area of images, skip if more than max_size square 

        """
        # to eval
        self.eval()
        
        # file writer
        if save_path is not None:
            self.writer = h5py.File(str(save_path), 'a')
        
        # dataloader
        dataloader = self.__dataloader__(dataset,)
        
        # L N D
        features, imids = [], []

        # time
        start_time = time.time()
        
        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='green', desc='extract locals'.rjust(15))):
            
            #
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img, **kwargs) 
            img  = self.__cuda__(img) 
            
            # extract locals 
            preds   = self.model.extract_locals(img, scales=scales, num_features=num_features, **kwargs)
            desc    = preds['features']

            # numpy
            features.append(self.__to_numpy__(desc))
            imids.append(np.full((desc.shape[0],), it))
            
            # write
            if hasattr(self, 'writer'):
                name = data['name'][0]
                self.__write__(name, desc)
            
            # clear cache  
            if it % 10 == 0:
                torch.cuda.empty_cache()
        
        # close writer    
        if hasattr(self, 'writer'):  
            self.writer.close()  
    
        # stack
        features    = np.vstack(features)
        ids         = np.hstack(imids)    
        
        # end time
        end_time = time.time() - start_time  
        
        #
        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')  
        
        #
        out = {
            'features':     features,
            'ids':          ids,
            'save_path':    save_path
            }
        
        return out
                                    
          
            
if __name__ == '__main__':
    
    
    logger = setup_logger(output=".", name="retrieval")
    DATA_DIR='/media/loc/ssd_5127/tmp/how/how_data/test/oxford5k/jpg'
    save_path='db.h5'
    
    dataset = ImagesListDataset(DATA_DIR, max_size=500)
    
    # feature_extractor = FeatureExtractor("resnet50_c4_gem_1024")
    feature_extractor = FeatureExtractor("resnet50_c4_how")

    scales = [0.707, 1.0, 1.424]
    
    global_feat, pth = feature_extractor.extract_global(dataset,    scales=scales, save_path=save_path)
    local_feat, pth = feature_extractor.extract_locals(dataset,     scales=scales, save_path=save_path)
    
    logger.info("features path", pth)
    logger.info("global features shape", global_feat.shape)
    logger.info("global features shape", local_feat.shape)

        
  