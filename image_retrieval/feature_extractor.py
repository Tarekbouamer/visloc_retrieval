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

from image_retrieval.utils.logging import setup_logger
from image_retrieval.datasets import ImagesListDataset
from image_retrieval.models import create_model, get_pretrained_cfg


# logger
import logging
logger = logging.getLogger("retrieval")

__DEBUG__ = False

    
class FeatureExtractorOptions(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor():
    """ Feature extraction 
        
        model_name:     str     model name from models factory
        dataset:                ImagesListDataset
        output:     str         path to save features as h5py format     
    """
    def __init__(self, model_name, dataset=None, output=None):
        super().__init__()
        
        # options
        self.options = FeatureExtractorOptions()
        self.cfg     = get_pretrained_cfg(model_name)
        
        # dataset        
        if dataset is None:
            raise ValueError(f'dataset is None Type {dataset}')
        
        self.dataset    = dataset
        self.dataloader = DataLoader(dataset, num_workers=1)
        
        # build  
        self.model  = create_model(model_name, pretrained=True)
        self.model = self.__cuda__(self.model)
        self.model.eval()

        # transform
        self.transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ])
        
        # writer
        self.output = output
        if output is not None:
            self.writer = h5py.File(str(output), 'a')
            
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
        
    def __prepare_input__(self, x):
        x = self.transform(x)
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        return x
    
    def __to_numpy__(self, x):
        if len(x.shape) > 1:
            x = x.squeeze(0)
            
        if x.is_cuda:
            x = x.cpu()
            
        return x.numpy()
    
    def __check_size__(self, x, min_size=200, max_size=1200):
        # too large (area)
        if not (x.size(-1) * x.size(-2) <= max_size * max_size):
            return True
        # too small
        if not (x.size(-1) >= min_size and x.size(-2) >= min_size):
            return True
        return False
            
    @torch.no_grad()     
    def extract(self, scales=[1.0], min_size=200, max_size=1200):
        
        features = np.empty(shape=(len(self.dataset), self.cfg['out_dim']))
        
        start_time = time.time()
        
        for it, data in enumerate(tqdm(self.dataloader, total=len(self.dataloader), colour='magenta', desc='extract'.rjust(15))):
            
            img , name, original_size = data['img'], data['img_name'][0], data["original_size"][0]
            
            # prepare inputs
            img  = self.__prepare_input__(img) 
            img  = self.__cuda__(img) 
            
            #
            desc = torch.zeros(self.cfg['out_dim'])
            
            #
            num_scales = 0. 
            
            for scale in scales:
                # scale
                if scale == 1.0:
                    img_s = img
                else:
                    img_s = functional.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
                
                # assert size within boundaries
                if self.__check_size__(img_s, min_size, max_size):
                    continue
                
                num_scales += 1.0
                 
                # extract
                desc_s = self.model(img_s).squeeze(0)
                desc_s = self.__to_numpy__(desc_s)   
                
                # accum
                desc += (1./num_scales) * desc_s
            #
            features[it] = desc.norm()
            
            if hasattr(self, 'writer'):
                self.__write__(name, desc)
                
            if it % 10 == 0:
                torch.cuda.empty_cache()
        #   
        if hasattr(self, 'writer'):  
            self.writer.close()  
        
        # 
        end_time = time.time() - start_time  
        logger.info(f'extraction done {end_time:.4} seconds saved {self.output}')
        
        return features, self.output
                                    
            
if __name__ == '__main__':
    
    logger = setup_logger(output=".", name="retrieval")

    DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
    DATA_DIR='/media/loc/ssd_5126/tmp/how/how_data/test/oxford5k/jpg'
    OUT='db.h5'
    
    dataset = ImagesListDataset(DATA_DIR, max_size=500)
    
    feature_extractor = FeatureExtractor("resnet50_c4_gem_1024", dataset=dataset, output=OUT)
    scales = [0.5, 0.707, 1.0, 1.414, 2.0]
    
    feat, pth = feature_extractor.extract(scales)
    
    print("features path", pth)
    print("feature shape", feat.shape)

        
  