import h5py

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm


from image_retrieval.datasets import ImagesListDataset

from image_retrieval.models.factory import create_model


# logger
import logging
logger = logging.getLogger("retrieval")



class FeatureExtractorOptions(object):
    pass


class FeatureExtractor():
    def __init__(self, model_name, dataset=None, output=None):
        super().__init__()
        
        # options
        self.options = FeatureExtractorOptions()
        
        if dataset is None:
            raise ValueError(f'dataset is None Type {dataset}')
        
        #      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build  
        self.model  = create_model(model_name, pretrained=True).to(device=self.device)
        self.model.eval()

        # dataset
        self.dataset = dataset
                
        # transform
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, 
                                     std=IMAGENET_DEFAULT_STD)])
        
        self.output = output
         
    @torch.no_grad()       
    def extract(self):
        
        for data in tqdm(self.dataset, total=len(self.dataset), colour='magenta', desc='extract'.rjust(15)):
            img     = data['img']
            name    = data['img_name']
            
            img  = self.transform(img).unsqueeze(0).to(self.device)
            desc = self.model(img).squeeze(0).cpu().numpy()
            
            if self.output is not None:
                with h5py.File(str(self.output), 'a') as fd:
                    try:
                        if name in fd:
                            del fd[name]
                        
                        grp = fd.create_group(name)
                        grp.create_dataset('desc', data=desc)

                    except OSError as error:                    
                        raise error       
                        
            
if __name__ == '__main__':
    
    DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
    OUT='db.h5'
    
    dataset = ImagesListDataset(DATA_DIR, max_size=1024)
    
    feature_extractor = FeatureExtractor("resnet50_c4_gem_1024", dataset=dataset, output=OUT)
    feature_extractor.extract()
    print(feature_extractor)

        
  