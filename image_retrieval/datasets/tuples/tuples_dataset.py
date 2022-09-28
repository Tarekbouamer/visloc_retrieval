from os import path
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.utils.data as data

from image_retrieval.datasets.misc import cid2filename

from .dataset import ImagesFromList, INPUTS
from .transform import ImagesTransform

# logger
import logging
logger = logging.getLogger("retrieval")

NETWORK_INPUTS = ["q", "p", "ns"]
All_INPUTS     = ["q", "p", "ns"]


class TuplesDataset(data.Dataset):
    """

    """

    def __init__(self, root_dir, name, mode, batch_size=1, num_workers=1, neg_num=5, query_size=2000, pool_size=20000, transform=None):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("Mode should be either train or val, passed as string"))
        
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

        elif name.startswith('gl'):
            
            # setting up paths
            ims_root = path.join(root_dir, 'train')
    
            # loading db
            db_fn = path.join(root_dir, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]

        else:
            raise(RuntimeError("Unkno wn dataset name!"))
        

        # initializing tuples dataset
        self.name = name
        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.clusters       = db['cluster']
        self.query_pool     = db['qidxs']
        self.positive_pool  = db['pidxs']
        

        # size of training subset for an epoch
        self.neg_num = neg_num
        
        self.query_size = min(query_size,   len(self.query_pool))
        self.pool_size  = min(pool_size,    len(self.images))

        self.query_indices = None
        self.positive_indices = None
        self.negative_indices = None

        # transform
        self.transform = transform
        
    def load_img_from_desc(self, img_desc):

        if path.exists(img_desc + ".png"):
            img_file = img_desc + ".png"
        elif path.exists(img_desc + ".jpg"):
            img_file = img_desc + ".jpg"
        elif path.exists(img_desc):
            img_file = img_desc
        else:
            raise IOError("Cannot find any image for id {} ".format(img_desc))

        return Image.open(img_file).convert(mode="RGB")

    def _load_query(self, item):
        
        query_img_desc = self.images[self.query_indices[item]]

        query_img = self.load_img_from_desc(query_img_desc)

        rec = self.transform(query_img)

        query_img.close()
 
        return rec["img"]

    def _load_positive(self, item):

        positive_img_desc = self.images[self.positive_indices[item]]

        positive_img = self.load_img_from_desc(positive_img_desc)

        rec = self.transform(positive_img)

        return rec["img"]

    def _load_negative(self, item):
        
        negatives = []
        
        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]

            negative_img = self.load_img_from_desc(negative_img_desc)

            rec = self.transform(negative_img)
            
            negative_img.close()

            # Append
            negatives.append(rec["img"])


        return negatives

    def __len__(self):
        return self.query_size

    def __getitem__(self, item):

        # query image
        query       = self._load_query(item)
        positive    = self._load_positive(item)
        negatives   = self._load_negative(item)

        target = torch.Tensor([-1, 1] + [0]*len(negatives))

        tuple = [query, positive] + negatives
        tuple = [item.unsqueeze_(0) for item in tuple]
             
        return tuple, target
    
    def create_epoch_tuples(self, cfg, model):

        logger.debug(f'creating tuples ({self.name}) for mode ({self.mode})')
        
        global_cfg  = cfg["global"]
        data_cfg    = cfg["dataloader"]
        aug_cfg     = cfg["augmentaion"]

        # Set model to eval mode
        if model.training:
            model.eval()

        # Select positive pairs
        idxs2qpool = torch.randperm(len(self.query_pool))[:self.query_size]

        self.query_indices      = [self.query_pool[i]       for i in idxs2qpool]
        self.positive_indices   = [self.positive_pool[i]    for i in idxs2qpool]

        # Select negative pairs
        idxs2images = torch.randperm(len(self.images))[:self.pool_size]

        # TODO: one Image only at the time 
        batch_size = 1
        
        with torch.no_grad():
            
            # Prepare query loader
            logger.debug('extracting descriptors for query images :')

            tf = ImagesTransform( max_size=data_cfg.getint("max_size"))
            
            query_data = ImagesFromList(root='', 
                                        images=[self.images[i] for i in self.query_indices], 
                                        transform=tf)
            
            query_dl = torch.utils.data.DataLoader(query_data, 
                                                   batch_size = batch_size, 
                                                   shuffle=False,
                                                   sampler=None, 
                                                   num_workers=self.num_workers, 
                                                   pin_memory=True
                                                   )
           
            # Extract query vectors
            qvecs = torch.zeros(len(self.query_indices), global_cfg.getint("global_dim")).cuda()

            for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(device="cuda", non_blocking=True) for k in INPUTS}

                pred = model(**batch, do_whitening=True)
                                
                qvecs[it * batch_size: (it+1) * batch_size, :] = pred

                del pred

            
            # Prepare negative pool data loader
            logger.debug('extracting descriptors for negative pool :')
            
            pool_data = ImagesFromList(root='', 
                                       images=[self.images[i] for i in idxs2images], 
                                       transform=tf)
            
            pool_dl = torch.utils.data.DataLoader( pool_data, 
                                                   batch_size = batch_size, 
                                                   shuffle=False,
                                                   sampler=None, 
                                                   num_workers=self.num_workers, 
                                                   pin_memory=True
                                                   )
            
            # Extract negative pool vectors
            poolvecs = torch.zeros(len(idxs2images),global_cfg.getint("global_dim")).cuda()

            for it, batch in tqdm(enumerate(pool_dl), total=len(pool_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(device="cuda", non_blocking=True) for k in INPUTS}

                pred = model(**batch, do_whitening=True)

                poolvecs[it * batch_size: (it+1) * batch_size, :] = pred
           
                del pred

            
            logger.debug('searching for hard negatives :')
            # Compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs, qvecs.t())
            scores, scores_indices = torch.sort(scores, dim=0, descending=True)

            average_negative_distance   = torch.tensor(0).float().cuda()  # for statistics
            negative_distance           = torch.tensor(0).float().cuda()  # for statistics

            # Selection of negative examples
            self.negative_indices = []

            for q in range(len(self.query_indices)):

                # Do not use query cluster those images are potentially positive
                qcluster = self.clusters[self.query_indices[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.neg_num:
                    potential = idxs2images[scores_indices[r, q]]
                    # take at most one image from the same cluster

                    if not self.clusters[potential] in clusters:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])

                        average_negative_distance += torch.pow(qvecs[q]-poolvecs[scores_indices[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        negative_distance += 1
                    r += 1
                self.negative_indices.append(nidxs)

            del scores
            avg_negative_l2 = (average_negative_distance/negative_distance).item()
            logger.info('average negative l2-distance = %f', avg_negative_l2)
        
        # stats
        stats = {
            "avg_negative_l2": avg_negative_l2
        }
        return   stats
