from os import path

from image_retrieval.datasets import ImagesFromList, ImagesTransform, INPUTS, ParisOxfordTestDataset

from torch.utils.data import DataLoader

# logger
import logging
logger = logging.getLogger("retrieval")


def build_paris_oxford_dataset(data_path, name_dataset, cfg):
    
    assert path.exists(data_path), logger.error("path: {data_path} does not exsists !!")
    
    logger.info(f'[{name_dataset}]: loading test dataset from {data_path}')
         
    db = ParisOxfordTestDataset(root_dir=data_path, name=name_dataset)
    
    # options 
    trans_opt = {   "max_size":     cfg["test"].getint("max_size")}
            
    dl_opt = {  "batch_size":   1, 
                "shuffle":      False, 
                "num_workers":  cfg["test"].getint("num_workers"), 
                "pin_memory":   True }  
    
    # query loader
    query_data  = ImagesFromList(root='', images=db['query_names'], bbxs=db['query_bbx'], transform=ImagesTransform(**trans_opt) )
    query_dl    = DataLoader( query_data, **dl_opt)
    
    # database loader
    db_data     = ImagesFromList( root='', images=db['img_names'], transform=ImagesTransform(**trans_opt) )  
    db_dl       = DataLoader(  db_data,  **dl_opt )
    
    # ground
    ground_truth = db['gnd']
    
    return query_dl, db_dl, ground_truth