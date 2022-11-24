from os import path

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]

def cid2filename(cid, prefix):
    """ Creates a training image path out of its CID name
        cid      : name of the image
        prefix   : root directory where images are saved
    """
    return path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)
