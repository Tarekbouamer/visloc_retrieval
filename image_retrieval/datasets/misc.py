import torch
from os import path

from torch.utils.data import DataLoader, default_collate


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    Args:
        batch: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None, None, None, None, None

    tuple, target = zip(*batch)

    tuple  = default_collate(tuple)
    target = default_collate(target)

    return tuple, target
    
    



def cid2filename(cid, prefix):
    """
        Creates a training image path out of its CID name

        Arguments
        ---------
        cid      : name of the image
        prefix   : root directory where images are saved

        Returns
        -------
        filename : full image filename
    """
    return path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)
