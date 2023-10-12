import random
from os import path

import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import INPUTS, ImagesFromList
from .transform import ImagesTransform

NETWORK_INPUTS = ["q", "p", "ns"]

_Ext = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']


class TuplesDataset(Dataset):
    """ TuplesDataset"""

    def __init__(self, data_path, name="tuples", cfg={}, mode="train", transform=None):

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)

        # name
        self.name = name

        # mode
        assert mode in ['train', 'val'], "mode must be either train or val"
        self.mode = mode

        # data_path
        self.data_path = data_path

        # images
        self.images = []

        # clusters and pool of images
        self.clusters = None
        self.query_pool = None
        self.positive_pool = None

        # sizes
        self.neg_num = cfg.data.neg_num
        self.query_size = cfg.data.query_size
        self.pool_size = cfg.data.pool_size

        # indices
        self.query_indices = None
        self.positive_indices = None
        self.negative_indices = None

        # transform
        self.transform = transform

    def build_dataset(self):
        """ Build dataset """
        raise NotImplementedError

    def __len__(self):
        """ Return the number of tuples"""
        return self.query_size

    def load_image(self, img_path):
        """ Load image"""

        # use
        for ext in _Ext:
            img_file = path.join(self.data_path, img_path)
            if path.exists(img_file):
                break
        try:
            image = Image.open(img_file).convert('RGB')
        except:
            logger.error(f'Error loading image: {img_file}')
            raise

        return image

    def get_query(self, item):
        """ Get a query image"""

        # query image
        query_path = self.images[self.query_indices[item]]
        query_img = self.load_image(query_path)

        # transform
        rec = self.transform(query_img)

        return rec["image"]

    def get_positives(self, item):
        """ Get positive images """

        # positive image
        idx = self.positive_indices[item]

        # sample
        idx = random.choice(idx) if isinstance(idx, list) else idx

        # positive image
        positive_img_desc = self.images[idx]
        positive_img = self.load_image(positive_img_desc)

        # transform
        rec = self.transform(positive_img)

        return rec["image"]

    def get_negatives(self, item):
        """ Get list of negative images"""

        negatives = []

        # negative images
        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]
            negative_img = self.load_image(negative_img_desc)

            # transform
            rec = self.transform(negative_img)

            # append
            negatives.append(rec["image"])

        return negatives

    def __getitem__(self, item):

        # query image
        query = self.get_query(item)

        # positive image
        positive = self.get_positives(item)

        # negative images
        negatives = self.get_negatives(item)

        # target label (-1 for query, 1 for positive, 0 for negative)
        target = torch.Tensor([-1, 1] + [0]*len(negatives))

        tuple = [query, positive] + negatives
        tuple = [item.unsqueeze_(0) for item in tuple]

        return tuple, target

    @torch.no_grad()
    def create_epoch_tuples(self, cfg, model):
        """ Create epoch tuples, Hard negative mining """

        logger.debug(f'Creating tuples ({self.name}) for mode ({self.mode})')

        # eval mode
        if model.training:
            model.eval()

        # select positive pairs
        idxs2qpool = torch.randperm(len(self.query_pool))[:self.query_size]

        self.query_indices = [self.query_pool[i] for i in idxs2qpool]
        self.positive_indices = [self.positive_pool[i] for i in idxs2qpool]

        # select negative pairs
        idxs2images = torch.randperm(len(self.images))[:self.pool_size]

        # options
        dl = {'batch_size':   1,  'num_workers':  4,
              'shuffle': False,   'pin_memory': True}
        tf = ImagesTransform(max_size=cfg.data.max_size)

        # Prepare data loader
        logger.debug('Extracting descriptors for query images :')

        query_data = ImagesFromList(data_path='', images_names=[self.images[i] for i in self.query_indices],
                                    transform=tf)
        query_dl = DataLoader(query_data, **dl)

        # Extract query vectors
        qvecs = torch.zeros(len(self.query_indices), model.dim).cuda()

        for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):
            batch = {k: batch[k].cuda(
                device="cuda", non_blocking=True) for k in INPUTS}

            pred = model(batch, do_whitening=True)
            qvecs[it] = pred["features"]

            del pred

        # Prepare negative pool data loader
        logger.debug('Extracting descriptors for negative pool :')

        pool_data = ImagesFromList(
            data_path='', images_names=[self.images[i] for i in idxs2images], transform=tf)
        pool_dl = DataLoader(pool_data, **dl)

        # Extract negative pool vectors
        poolvecs = torch.zeros(len(idxs2images), model.dim).cuda()

        for it, batch in tqdm(enumerate(pool_dl), total=len(pool_dl)):
            batch = {k: batch[k].cuda(
                device="cuda", non_blocking=True) for k in INPUTS}

            pred = model(batch, do_whitening=True)
            poolvecs[it] = pred["features"]

            del pred

        logger.debug('Searching for hard negatives :')

        # Compute dot product scores and ranks on GPU
        scores = torch.mm(poolvecs, qvecs.t())
        scores, scores_indices = torch.sort(scores, dim=0, descending=True)

        average_negative_distance = torch.tensor(0).float().cuda()
        negative_distance = torch.tensor(0).float().cuda()

        # selection of negative examples
        self.negative_indices = []

        for q in range(len(self.query_indices)):

            # do not use query cluster those images are potentially positive
            # take at most one image from the same cluster

            qcluster = self.clusters[self.query_indices[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            while len(nidxs) < self.neg_num:
                potential = idxs2images[scores_indices[r, q]]

                if self.clusters[potential] not in clusters:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])

                    average_negative_distance += torch.pow(
                        qvecs[q]-poolvecs[scores_indices[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    negative_distance += 1
                r += 1

            self.negative_indices.append(nidxs)

        del scores

        avg_negative_l2 = (average_negative_distance /
                           negative_distance).item()
        logger.info(
            f'Average negative l2-distance = {avg_negative_l2:.3}', )

        # stats
        stats = {
            "avg_negative_l2": avg_negative_l2}

        return stats
