from os import path
import torch.utils.data as data
from PIL import Image

import pickle
from image_retrieval.datasets.misc import cid2filename

import torch

INPUTS = ["img"]

class ISSDataset(data.Dataset):

    def __init__(self, root_dir, name, mode, transform=None):
        super(ISSDataset, self).__init__()
        
        self.root_dir = root_dir
        self.name = name
        self.mode = mode
        
        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("Mode should be either train or val, passed as string"))
        
        # load sequences
        if name.startswith('retrieval-SfM'):

            # setting up paths
            db_root = path.join(root_dir, 'train', name)
            ims_root = path.join(db_root, 'ims')
    
            # loading db
            db_fn = path.join(db_root, '{}.pkl'.format(name))
            
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # get images full path
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        

        if len(self.images) == 0:
            raise(RuntimeError("Empty Dataset !"))
        
        # classes
        self.classes = db['cluster']

        # transforms
        self.transform = transform

    def _load_item(self, item):
        img_desc = self.images[item]

        if path.exists(img_desc + ".png"):
            img_file = img_desc + ".png"
        elif path.exists(img_desc + ".jpg"):
            img_file = img_desc + ".jpg"
        elif path.exists(img_desc):
            img_file = img_desc
        else:
            raise IOError("Cannot find any image for id {} ".format(img_desc))

        img = Image.open(img_file).convert(mode="RGB")

        return img, img_desc

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # get image
        img, img_desc = self._load_item(item)

        # classes
        cls_item = self.classes[item]
        
        rec = self.transform(img)

        size = (img.size[1], img.size[0])

        img.close()

        rec["idx"] = item
        rec["img_desc"] = img_desc
        rec["cls"] = torch.tensor(cls_item).long()

        rec["size"] = size
        
        return rec

