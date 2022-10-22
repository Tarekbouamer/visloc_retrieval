import pickle
from os import path

from retrieval.datasets.misc import cid2filename

TEST_DATASETS = ["val_eccv20", 
                 'oxford5k', 'paris6k', 
                 'roxford5k', 'rparis6k']

_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def ParisOxfordTestDataset(root_dir, name=None):
    """
        Paris Oxford dataset
          
        source:     https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
                    http://cmp.felk.cvut.cz/revisitop/
                    https://github.com/gtolias/how/
    """


    if name not in TEST_DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(name))
    
    meta = {}
    
    meta['_ext']          = '.jpg'    
    meta['dataset']       = name   
    
    if name == "val_eccv20":
        db_root = path.join(root_dir, 'train', 'retrieval-SfM-120k')
        pkl_path = path.join(db_root, "retrieval-SfM-120k-val-eccv2020.pkl")
        
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        
        ims_root = path.join(db_root, 'ims')
        images   = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
       
        meta['data_path']    = path.join(db_root)
        meta['images_path']   = path.join(meta['data_path'], 'jpg')

        meta['n_img']       = len(db['cids'])
        meta['n_query']     = len(db['qidx'])
        meta['img_names']   = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        meta['query_names'] = [images[x] for x in db['qidx']]
        meta['gnd']         = db['gnd']      
        meta['query_bbx']   = None
        
    else:
        db_root = path.join(root_dir, 'test', name)
        pkl_path = path.join(db_root,'gnd_{}.pkl'.format(name))
        
        print
        
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        
        meta['data_path']    = path.join(db_root)
        meta['images_path']   = path.join(meta['data_path'], 'jpg')
        
        meta['n_img']       = len(db['imlist'])
        meta['n_query']     = len(db['qimlist'])
        meta['img_names']   = [path.join(meta['images_path'], item + meta['_ext']) for item in db['imlist']]
        meta['query_names'] = [path.join(meta['images_path'], item + meta['_ext']) for item in db['qimlist']]
        meta['gnd']         = db['gnd']      
        meta['query_bbx']   = [db['gnd'][item]['bbx'] for item in range(meta['n_query'])]

    meta['pkl_path']      = pkl_path

    return meta