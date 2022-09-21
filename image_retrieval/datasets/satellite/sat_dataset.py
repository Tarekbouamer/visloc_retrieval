from os import path
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.utils.data as data


from image_retrieval.datasets.generic.generic import ImagesFromList, ImagesTransform, INPUTS
from image_retrieval.datasets.misc import cid2filename

NETWORK_INPUTS = ["q", "p", "ns"]
All_INPUTS     = ["q", "p", "ns"]

default_cities = {
    
    'cls' : [                 
                "00",
                "01",
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
                "15"                
                ],
    
    'train' : [                 
                # "00",
                "01",
                "03",
                "04", 
                "05", 
                "06", 
                "07", 
                "08", 
                # "09", 
                "10", 
                "11", 
                "12", 
                "13", 
                "14", 
                "15"
                ],
    
    'val'   : [
                "08", 
                "13", 
                ],
    
    'test'  : [
                "08", 
                "13", 
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

                print("=====> {}".format(city))
                
                q_offset  = len(self.qImages)
                db_offset = len(self.images)
                
                # Read pkl file of each scene 
                pkl_path = path.join(root_dir, city, "meta.pkl") 
                    
                if not path.exists(pkl_path):
                    print("Path not found", pkl_path)
                    exit(0)
                
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    
                # when GPS is available
                if mode in ['train', 'val']:
                
                    # concatenate images from with their full path 
                    self.qImages.extend(    [ path.join(root_dir, city, ext) for ext in data["q_imgs"]  ]   )
                    self.images.extend(     [ path.join(root_dir, city, ext) for ext in data["db_imgs"] ]   )
                    
                    # 
                    self.qIdx.extend(       [     q   + q_offset    for q   in data["q_idx"]    ]   )
                    self.pIdx.extend(       [     p   + db_offset   for p   in data["p_geo_idx"]    ]   )
                    self.nonNegIdx.extend(  [     non + db_offset   for non in data["non_idx"]  ]   )
                    #
                    assert len(self.qIdx) == len(self.pIdx) == len(self.nonNegIdx)

                elif mode in ['test']:

                    # load images
                    self.qImages.extend(    [ path.join(root_dir, city, ext) for ext in data["q_imgs"]  ]   )
                    self.images.extend(   [ path.join(root_dir, city, ext) for ext in data["db_imgs"] ]   )
                    
                    # 
                    self.qIdx.extend(       [     q   + q_offset    for q   in data["q_idx"]    ]   )

        else:
            raise(RuntimeError("Unknown dataset name!"))
        

        # initializing tuples dataset
        self.name = name
        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # size of training subset for an epoch
        self.neg_num = neg_num
        
        self.query_size = min(query_size,   len(self.qIdx))
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

    def _load_query(self, item, out):

        query_img_desc = self.qImages[self.query_indices[item]]

        query_img = self.load_img_from_desc(query_img_desc)

        rec = self.transform(query_img)

        query_size = (query_img.size[1], query_img.size[0])

        query_img.close()

        out["q"]        = rec["img"]
        out["q_desc"]   = query_img_desc
        out["q_size"]   = query_size
        
        return out, rec["img"]

    def _load_positive(self, item, out):
    
        positive_img_desc = self.images[self.positive_indices[item]]

        positive_img = self.load_img_from_desc(positive_img_desc)

        rec = self.transform(positive_img)

        query_size = (positive_img.size[1], positive_img.size[0])

        positive_img.close()

        out["p"]        = rec["img"]
        out["p_desc"]   = positive_img_desc
        out["p_size"]   = query_size

        return out, rec["img"]

    def _load_negative(self, item, out):
        n_images = []
        n_images_desc = []
        n_query_size = []

        negatives = []
        
        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]

            negative_img = self.load_img_from_desc(negative_img_desc)

            rec = self.transform(negative_img)
            
            query_size = (negative_img.size[1], negative_img.size[0])
            
            negative_img.close()

            # Append
            n_images.append(rec["img"])
            n_images_desc.append(negative_img_desc)
            n_query_size.append(query_size)
            
            negatives.append(rec["img"])

        # Re make output
        out["ns"]       = n_images
        out["ns_desc"]  = n_images_desc
        out["ns_size"]  = n_query_size

        return out, negatives

    def __len__(self):
        return self.query_size

    def __getitem__(self, item):

        out = dict()
        out["idx"] = item

        # query image
        out, query = self._load_query(item, out)

        # positive image
        out, positive = self._load_positive(item, out)

        # negative images
        out, negatives = self._load_negative(item, out)

        out["neg_nums"] = len(out["ns"])
        
        # target if needed
        out["target"] = torch.Tensor([-1, 1] + [0]*len(out["ns"]))

        #
        output = [query]
        output.extend([positive])
        output.extend(negatives)

        # return out, output, target, out["neg_nums"]
        return out
    
    def create_epoch_tuples(self, model, log_info, log_debug, **varargs):

        log_debug('Creating tuples for an epoch of {%s}--{%s}', self.name, self.mode)

        data_config = varargs["data_config"]

        # Set model to eval mode
        model.eval()

        # Select random indices
        indices = torch.randperm(len(self.qIdx))[:self.query_size]

        # query
        q_indices   = [self.qIdx[i]         for i in indices]
        
        # positive set
        p_indices   = [self.pIdx[i]         for i in indices]
        p_indices   = np.unique([i for idx in p_indices for i in idx])

        # negative
        n_indices   = np.random.choice(len(self.images), self.pool_size, replace=False)
        non_indices = [self.nonNegIdx[i]    for i in indices]
        n_indices   = n_indices[np.in1d(n_indices, np.unique([i for idx in non_indices for i in idx]), invert=True)]


        # TODO: one Image only at the time 
        batch_size = 1
        tf_opt = {
                "shortest_size":    data_config.getint("train_shortest_size"), 
                "longest_max_size": data_config.getint("train_longest_max_size"),
                "rgb_mean":         data_config.getstruct("rgb_mean"),
                "rgb_std":          data_config.getstruct("rgb_std")
                }
            
        dl_opt = {
                "batch_size": batch_size,
                "shuffle":False,
                "sampler":None,
                "num_workers":self.num_workers,
                "pin_memory":True
            }

        # Transform    
        transform    = ImagesTransform(**tf_opt)
        
        # Dataloaders
        q_dl     = data.DataLoader( ImagesFromList(root='',  images=[self.qImages[i] for i in q_indices],   transform=transform), **dl_opt)
        p_dl     = data.DataLoader( ImagesFromList(root='',  images=[self.images[i]  for i in p_indices],   transform=transform), **dl_opt)
        n_dl     = data.DataLoader( ImagesFromList(root='',  images=[self.images[i]  for i in n_indices],   transform=transform), **dl_opt)

        # 
        self.query_indices      = []
        self.positive_indices   = []
        self.negative_indices   = []
        
        with torch.no_grad():
            
            # Query
            log_debug('Extracting descriptors for query images :')

            qvecs = torch.zeros(varargs["output_dim"], len(q_indices)).cuda()
            
            for it, batch in tqdm(enumerate(q_dl), total=len(q_dl)):
                # Upload batch
                batch   = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}
                pred    = model(**batch, do_prediction=True, do_whitening=True)
                qvecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]
                del pred


            # Positves             
            log_debug('Extracting descriptors for positive images :')

            pvecs = torch.zeros(varargs["output_dim"], len(p_indices)).cuda()
            
            for it, batch in tqdm(enumerate(p_dl), total=len(p_dl)):
                # Upload batch
                batch   = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}
                pred    = model(**batch, do_prediction=True, do_whitening=True)
                pvecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]
                del pred
                                                            
            # Negatives
            log_debug('Extracting descriptors for negative pool :')
            
            nvecs = torch.zeros(varargs["output_dim"], len(n_indices)).cuda()
            
            for it, batch in tqdm(enumerate(n_dl), total=len(n_dl)):
                # Upload batch
                batch   = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}
                pred    = model(**batch, do_prediction=True, do_whitening=True)
                nvecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]
                del pred
                
                

            log_debug('Searching for hard negatives :')
            
            # Compute dot product scores and ranks on GPU
            pScores = torch.mm(pvecs.t(), qvecs)
            pScores, pRanks = torch.sort(pScores, dim=0, descending=True)
                
            
            #
            nScores = torch.mm(nvecs.t(), qvecs)
            nScores, nRanks = torch.sort(nScores, dim=0, descending=True)

            # convert to cpu and numpy
            pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
            nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()
            
            average_negative_distance   = torch.tensor(0).float().cuda()  # for statistics
            negative_distance           = torch.tensor(0).float().cuda()  # for statistics

            for q in range(len(q_indices)):
                
                qidx = q_indices[q]

                # find positive idx for this query (cache idx domain)
                cached_pidx = np.where(np.in1d(p_indices, self.pIdx[qidx]))
                pidx        = np.where(np.in1d(pRanks[:, q], cached_pidx))
 
                # take the closest positve
                dPos = pScores[pidx, q][0][0]
                
                # get distances to all negatives
                dNeg = nScores[:, q]
                
                # how much are they violating
                loss = dPos - dNeg + self.margin ** 0.5
                violatingNeg = 0 < loss
                
                # if less than nNeg are violating then skip this query
                if np.sum(violatingNeg) <= self.neg_num:
                    continue

                # select hardest negatives
                hardest_negIdx = np.argsort(loss)[:self.neg_num]

                # select the hardest negatives
                cached_hardestNeg = nRanks[hardest_negIdx, q]

                # select the closest positive (back to cache idx domain)
                cached_pidx = pRanks[pidx, q][0][0]

                # transform back to original index (back to original idx domain)
                qidx = self.qIdx[qidx]
                pidx = p_indices[cached_pidx]
                hardestNeg = n_indices[cached_hardestNeg]
    
                self.query_indices.append(qidx)
                self.positive_indices.append(pidx)
                self.negative_indices.append(hardestNeg)
                
                


                
                
                
                
                # nidxs = []
                # r = 0
                # while len(nidxs) < self.neg_num:
                #     potential = idxs2images[scores_indices[r, q]]
                #     # take at most one image from the same cluster

                #     nidxs.append(potential)

                #     average_negative_distance += torch.pow(qvecs[:,q]-poolvecs[:,scores_indices[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                #     negative_distance += 1
                #     r += 1
                
                # self.negative_indices.append(nidxs)

            # del scores
            # log_info('Average negative l2-distance = %f', average_negative_distance/negative_distance)

        return (average_negative_distance/negative_distance).item()  # return average negative l2-distance
