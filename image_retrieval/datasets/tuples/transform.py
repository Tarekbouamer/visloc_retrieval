from PIL import Image

import  torchvision.transforms as transforms

from  timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ImagesTransform:

    def __init__(self, max_size, preprocessing=None, augmentation=None, postprocessing=None,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD):
        
        self.max_size = max_size
        self.mean   = mean
        self.std    = std
        
        # preprocessing 
        if preprocessing:
            self.preprocessing  = preprocessing
        
        # augmentation
        if augmentation:
            self.augmentation   = augmentation

        # to tensor
        if postprocessing:
            self.postprocessing = postprocessing
        else:
            self.postprocessing = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, img):
        
        #
        if hasattr(self, 'preprocessing'):
            img = self.preprocessing(img)
        else:
            img.thumbnail((self.max_size, self.max_size), Image.BILINEAR)        

        #
        if hasattr(self, 'augmentation'):
            img = self.augmentation(img)
            
        #
        img = self.postprocessing(img)           

        return dict(img=img)