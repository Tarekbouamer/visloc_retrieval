import pickle
from os import path

import numpy as np
import torch
import torch.utils.data as data
from loguru import logger
from PIL import Image
from tqdm import tqdm

NETWORK_INPUTS = ["q", "p", "ns"]
All_INPUTS = ["q", "p", "ns"]

default_cities = {
    'train': [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ],

    'val': [
        "03",
        "09",
        "16",
    ],

    'test': [
        "03",
        "09",
        "16",
    ]
}


class SatDataset(data.Dataset):
    """

    """

    def __init__(self, root_dir, name, mode, batch_size=1, num_workers=1, neg_num=5, query_size=2000, pool_size=20000, transform=None, margin=0.5):

        assert mode in ('train', 'val', 'test')

        self.qImages = []
        self.images = []

        self.qIdx = []
        self.pIdx = []
        self.nonNegIdx = []

        self.margin = margin

        if name.startswith('SAT'):

            # cities
            self.cities = default_cities[mode]

            for city in self.cities:

                print(f"=====> {city}")

                q_offset = len(self.qImages)
                db_offset = len(self.images)

                # Read pkl file of each scene
                pkl_path = path.join(root_dir, city, "meta_aug.pkl")

                if not path.exists(pkl_path):
                    logger.error("Path not found", pkl_path)
                    exit(0)

                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                # when GPS is available
                if mode in ['train', 'val']:

                    # concatenate images from with their full path
                    self.qImages.extend(
                        [path.join(root_dir, city, ext) for ext in data["q_imgs"]])
                    self.images.extend([path.join(root_dir, city, ext)
                                       for ext in data["db_imgs"]])

                    #
                    self.qIdx.extend([q + q_offset for q in data["q_idx"]])
                    self.pIdx.extend(
                        [p + db_offset for p in data["p_geo_idx"]])
                    self.nonNegIdx.extend(
                        [non + db_offset for non in data["non_idx"]])

                # elif mode in ['test']:

                #     # load images
                #     self.qImages.extend(    [ path.join(root_dir, city, ext) for ext in data["q_imgs"]  ]   )
                #     self.images.extend(   [ path.join(root_dir, city, ext) for ext in data["db_imgs"] ]   )

                #     #
                #     self.qIdx.extend(       [     q   + q_offset    for q   in data["q_idx"]    ]   )

        else:
            raise (RuntimeError("Unknown dataset name!"))

        assert len(self.qIdx) == len(self.pIdx) == len(self.nonNegIdx)

        # initializing tuples dataset
        self.name = name
        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers

        # size of training subset for an epoch
        self.neg_num = neg_num

        self.query_size = min(query_size,   len(self.qIdx))
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

        query_img_desc = self.qImages[self.query_indices[item]]
        query_img = self.load_img_from_desc(query_img_desc)

        rec = self.transform(query_img)
        query_img.close()

        return rec["image"]

    def _load_positive(self, item):

        positive_img_desc = self.images[self.positive_indices[item]]
        positive_img = self.load_img_from_desc(positive_img_desc)

        rec = self.transform(positive_img)
        positive_img.close()

        return rec["image"]

    def _load_negative(self, item):

        negatives = []
        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]
            negative_img = self.load_img_from_desc(negative_img_desc)

            rec = self.transform(negative_img)
            negative_img.close()

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

    def create_epoch_tuples(self,  cfg, model):

        logger.info(
            'Creating tuples for an epoch of {%s}--{%s}', self.name, self.mode)

        # Set model to eval mode
        if model.training:
            model.eval()

        # Select random indices
        indices = torch.randperm(len(self.qIdx))[:self.query_size]

        # query set
        q_indices = [self.qIdx[i] for i in indices]

        # positive set
        p_indices = [self.pIdx[i] for i in indices]
        p_indices = np.unique([i for idx in p_indices for i in idx])

        # negative set
        n_indices = np.random.choice(
            len(self.images), self.pool_size, replace=False)
        non_indices = [self.nonNegIdx[i] for i in indices]
        n_indices = n_indices[np.in1d(n_indices, np.unique(
            [i for idx in non_indices for i in idx]), invert=True)]

        # options
        batch_size = 1
        dl_opt = {
            "batch_size": batch_size,
            "shuffle": False,
            "sampler": None,
            "num_workers": self.num_workers,
            "pin_memory": True
        }

        # Transform
        tf = ImagesTransform(max_size=cfg.dataloader.max_size)

        # Dataloaders
        q_dl = data.DataLoader(ImagesFromList(root='',  images=[
                               self.qImages[i] for i in q_indices],   transform=tf), **dl_opt)
        p_dl = data.DataLoader(ImagesFromList(root='',  images=[
                               self.images[i] for i in p_indices],   transform=tf), **dl_opt)
        n_dl = data.DataLoader(ImagesFromList(root='',  images=[
                               self.images[i] for i in n_indices],   transform=tf), **dl_opt)

        #

        self.query_indices = []
        self.positive_indices = []
        self.negative_indices = []

        with torch.no_grad():

            # extract queries
            logger.debug('extracting descriptors for query images :')
            qvecs = torch.zeros(len(q_dl), cfg.body.out_dim).cuda()
            for it, batch in tqdm(enumerate(q_dl), total=len(q_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(
                    device="cuda", non_blocking=True) for k in INPUTS}
                pred = model(**batch, do_whitening=True)
                qvecs[it * batch_size: (it+1) * batch_size, :] = pred
                del pred

            # extract positives
            logger.debug('extracting descriptors for positive images :')

            pvecs = torch.zeros(
                len(p_dl), cfg.body.out_dim).cuda()

            for it, batch in tqdm(enumerate(p_dl), total=len(p_dl)):
                # Upload batch
                batch = {k: batch[k].cuda(
                    device="cuda", non_blocking=True) for k in INPUTS}
                pred = model(**batch, do_whitening=True)
                pvecs[it * batch_size: (it+1) * batch_size, :] = pred
                del pred

            # extract negatives
            logger.debug('Extracting descriptors for negative pool :')

            nvecs = torch.zeros(
                len(n_dl), cfg.body.out_dim).cuda()

            for it, batch in tqdm(enumerate(n_dl), total=len(n_dl)):
                # Upload batch
                batch = {k: batch[k].cuda(
                    device="cuda", non_blocking=True) for k in INPUTS}
                pred = model(**batch, do_whitening=True)
                nvecs[it * batch_size: (it+1) * batch_size, :] = pred
                del pred

            logger.debug('Searching for hard negatives :')

            # Compute dot product scores and ranks on GPU
            pScores = torch.mm(qvecs, pvecs.t())
            pScores, pRanks = torch.sort(pScores, dim=-1, descending=True)

            #
            nScores = torch.mm(qvecs, nvecs.t())
            nScores, nRanks = torch.sort(nScores, dim=-1, descending=True)

            # convert to cpu and numpy
            pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
            nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

            torch.tensor(0).float().cuda()  # for statistics
            torch.tensor(0).float().cuda()  # for statistics

            for q in range(len(q_indices)):

                qidx = indices[q]

                # find positive idx for this query (cache idx domain)
                cached_p_indices = np.where(
                    np.in1d(p_indices, self.pIdx[qidx]))
                pidx = np.where(np.in1d(pRanks[q, :], cached_p_indices))

                # take the closest positve
                dPos = pScores[q, pidx][0][0]

                # get distances to all negatives
                dNeg = nScores[q, :]

                # how much are they violating
                loss = dPos - dNeg + self.margin ** 0.5
                violatingNeg = 0 < loss

                # if less than nNeg are violating then skip this query
                if np.sum(violatingNeg) <= self.neg_num:
                    continue

                # select hardest negatives
                hardest_negIdx = np.argsort(loss)[:self.neg_num]

                # select the hardest negatives
                cached_hardestNeg = nRanks[q, hardest_negIdx]

                # select the closest positive (back to cache idx domain)
                cached_p_indices = pRanks[q, pidx][0][0]

                # transform back to original index (back to original idx domain)
                qidx = self.qIdx[qidx]
                pidx = p_indices[cached_p_indices]
                hardestNeg = n_indices[cached_hardestNeg]

                #
                self.query_indices.append(qidx)
                self.positive_indices.append(pidx)
                self.negative_indices.append(hardestNeg)

        return {}
