from os import path
from  torch.utils.data import Dataset

from PIL        import Image, ImageFile
from pathlib    import Path

INPUTS = ["image"]


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']


class ImagesFromList(Dataset):
    """ImagesFromList
        generic dataset from list of images
    """
    def __init__(self, root, images=None, bbxs=None, transform=None):
        
        if images is not None:
            images_fn = [path.join(root, images[i]) for i in range(len(images))]
        else:
            # Load images
            images_fn = []
            for ext in _EXT:
                images_fn += list(Path(root).glob('**/'+ ext)) 

        if len(images_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))

        self.root = root
        self.images = images
        
        self.images_fn = images_fn
        
        self.bbxs = bbxs
        self.transform = transform

    def __len__(self):
        return len(self.images_fn)

    def load_img(self, img_path):
        
        # for truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True     
        
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
          
        return image
    
    def get_name(self,  _path):
        return path.basename(_path)
    
    def __getitem__(self, item):

        # image path and image name
        img_path    = self.images_fn[item]
        img_name    = self.get_name(img_path)
        
        # load image
        image = self.load_img(img_path)
        
        # crop image if box exsists 
        if self.bbxs is not None:
            image = image.crop(self.bbxs[item])

        # transform
        if self.transform is not None:
            out = self.transform(image)

        #
        out['name'] = img_name
        
        #
        return out