import pickle
import random
from os import path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from retrieval.datasets.misc import cid2filename

from .dataset import INPUTS, ImagesFromList
from .transform import ImagesTransform

NETWORK_INPUTS = ["q", "p", "ns"]
All_INPUTS = ["q", "p", "ns"]


def load_sfm_db_images(root_dir, name, mode):
    # setting up paths
    db_root = path.join(root_dir, 'train', name)
    ims_root = path.join(db_root, 'ims')
    db_fn = path.join(db_root, f'{name}.pkl')

    with open(db_fn, 'rb') as f:
        db = pickle.load(f)[mode]

    # get images full path
    images = [cid2filename(db['cids'][i], ims_root)
              for i in range(len(db['cids']))]

    return images, db


def load_gl_db_images(root_dir, name, mode):
    # root
    root_dir = path.join(root_dir, name)
    ims_root = path.join(root_dir, 'train')
    db_fn = path.join(root_dir, f'{name}.pkl')

    if not path.exists(db_fn):
        return [], {}

    with open(db_fn, 'rb') as f:
        db = pickle.load(f)[mode]

    # setting fullpath for images
    images = [path.join(ims_root, db['cids'][i]+'.jpg')
              for i in range(len(db['cids']))]

    return images, db


class TuplesDataset(Dataset):
    """ TuplesDataset
            mode:           train or val 
            neg_num:        number of negative examples
            query_size:     number of query images per epoch
            pool_size:      number of pool or databse images randomly selected from the pool
            transform:      sequnence of transformations (preprocessing, augmentation, ...) applied on loaded images
    """

    def __init__(self, root_dir, name, mode, neg_num=5, query_size=2000, pool_size=20000, transform=None):

        assert mode in ['train', 'val'], RuntimeError(
            "Mode should be either train or val, passed as string")

        if name.startswith('retrieval-SfM'):
            self.images, db = load_sfm_db_images(root_dir, name, mode)
        elif name.startswith('gl'):
            self.images, db = load_gl_db_images(root_dir, name, mode)
        else:
            raise (RuntimeError("Unkno wn dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode

        # indices
        self.clusters = db['cluster']
        self.query_pool = db['qidxs']
        self.positive_pool = db['pidxs']

        # size of training subset for an epoch
        self.neg_num = max(neg_num, 1)
        self.query_size = min(query_size,   len(self.query_pool))
        self.pool_size = min(pool_size,    len(self.images))

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
            raise IOError(f"Cannot find any image for id {img_desc} ")

        return Image.open(img_file).convert(mode="RGB")

    def _load_query(self, item):

        query_img_desc = self.images[self.query_indices[item]]
        query_img = self.load_img_from_desc(query_img_desc)

        rec = self.transform(query_img)

        return rec["image"]

    def _load_positive(self, item):

        idx = self.positive_indices[item]

        if isinstance(idx, list):
            idx = random.choice(idx)

        positive_img_desc = self.images[idx]
        positive_img = self.load_img_from_desc(positive_img_desc)

        rec = self.transform(positive_img)

        return rec["image"]

    def _load_negative(self, item):

        negatives = []

        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]
            negative_img = self.load_img_from_desc(negative_img_desc)

            rec = self.transform(negative_img)

            # Append
            negatives.append(rec["image"])

        return negatives

    def __len__(self):
        return self.query_size

    def __getitem__(self, item):

        # query image
        query = self._load_query(item)
        positive = self._load_positive(item)
        negatives = self._load_negative(item)

        target = torch.Tensor([-1, 1] + [0]*len(negatives))

        tuple = [query, positive] + negatives
        tuple = [item.unsqueeze_(0) for item in tuple]

        return tuple, target

    def create_epoch_tuples(self, cfg, model):
        """ Refresh dataset and search for new tuples via hard mining 
        """

        logger.debug(f'creating tuples ({self.name}) for mode ({self.mode})')

        # set model to eval mode
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
        tf = ImagesTransform(max_size=cfg.dataloader.max_size)

        with torch.no_grad():

            # Prepare data loader
            logger.debug('extracting descriptors for query images :')

            query_data = ImagesFromList(root='', images=[self.images[i] for i in self.query_indices],
                                        transform=tf)
            query_dl = DataLoader(query_data, **dl)

            # Extract query vectors
            qvecs = torch.zeros(len(self.query_indices),
                                model.dim).cuda()

            for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):
                batch = {k: batch[k].cuda(
                    device="cuda", non_blocking=True) for k in INPUTS}

                pred = model(batch, do_whitening=True)
                qvecs[it] = pred["features"]

                del pred

            # Prepare negative pool data loader
            logger.debug('extracting descriptors for negative pool :')

            pool_data = ImagesFromList(
                root='', images=[self.images[i] for i in idxs2images], transform=tf)
            pool_dl = DataLoader(pool_data, **dl)

            # Extract negative pool vectors
            poolvecs = torch.zeros(len(idxs2images), model.dim).cuda()

            for it, batch in tqdm(enumerate(pool_dl), total=len(pool_dl)):
                batch = {k: batch[k].cuda(
                    device="cuda", non_blocking=True) for k in INPUTS}

                pred = model(batch, do_whitening=True)
                poolvecs[it] = pred["features"]

                del pred

            logger.debug('searching for hard negatives :')

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
                f'average negative l2-distance = {avg_negative_l2:.3}', )

        # stats
        stats = {
            "avg_negative_l2": avg_negative_l2}

        return stats
