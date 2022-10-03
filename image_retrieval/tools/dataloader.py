from distutils.command.build import build
import numpy as np
import logging

from  torch.utils.data import DataLoader, SubsetRandomSampler
from  timm.data import create_transform
from  timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# image_retrieval
from image_retrieval.datasets.tuples import TuplesDataset, ImagesFromList, ImagesTransform 
from image_retrieval.datasets.satellite import SatDataset

from image_retrieval.datasets.misc  import collate_tuples

# logger
import logging
logger = logging.getLogger("retrieval")  

def build_dataset(args, cfg, transform, mode='train'):
    
    data_cfg    = cfg["dataloader"]
    test_cfg    = cfg["test"]
    
    data_opt = {    "neg_num": data_cfg.getint("neg_num"),
                    "batch_size": data_cfg.getint("batch_size"),
                    "num_workers": data_cfg.getint("num_workers")
                }
    # sfm
    if data_cfg.get("dataset") in ["retrieval-SfM-120k", "gl18"] :
        
        query_size  = data_cfg.getint("query_size")     if mode == 'train'  else float('inf')
        pool_size   = data_cfg.getint("pool_size")      if mode == 'train'    else float('inf')

        train_db = TuplesDataset(root_dir=args.data,
                                name=data_cfg.get("dataset"),
                                mode=mode,
                                query_size=query_size,
                                pool_size=pool_size,
                                transform=transform,
                                **data_opt) 
        
    
    elif  data_cfg.get("dataset") == "SAT" :
        
        train_db = SatDataset(root_dir=args.data,
                                name=data_cfg.get("dataset"),
                                mode=mode,
                                query_size=data_cfg.getint("query_size"),
                                pool_size=data_cfg.getint("pool_size"),
                                transform=transform,
                                **data_opt)
        
    
    return train_db  
    
def build_sample_dataloader(cfg, images):

    #
    num_samples = cfg["global"].getint("num_samples")
    
    if num_samples > len(images):
        num_samples = len(images)
    
    logger.debug(f"sample dataset:  {num_samples}")

    #
    sampler     = SubsetRandomSampler(np.random.choice(len(images), num_samples, replace=False))
    
    # transform
    transform = build_transforms(cfg)["test"]
             
    sample_dl   = DataLoader(dataset=ImagesFromList("", images, transform=transform),
                            num_workers=cfg["dataloader"].getint("num_workers"), 
                            pin_memory=True,
                            sampler=sampler)
    
    return sample_dl

def build_train_dataloader(args, cfg):
    data_cfg    = cfg["dataloader"]
    test_cfg    = cfg["test"]
    
    logger.info("build train dataloader")
    
    # Options
    dl_opt = {  "batch_sampler": None,          "batch_size":data_cfg.getint("batch_size"),
                "collate_fn":collate_tuples,    "pin_memory":True,
                "num_workers":data_cfg.getint("num_workers"), "shuffle":True, "drop_last":True}
    

    # transforms
    # transform = build_transforms(cfg)["train_aug"]
    transform = build_transforms(cfg)["train"]
    
    # dataset
    train_db = build_dataset(args, cfg, transform, mode='train')

    
    # loader
    train_dl = DataLoader(train_db, **dl_opt)
    
    return train_dl

def build_val_dataloader(args, cfg):
    data_cfg    = cfg["dataloader"]
    
    logger.info("build val dataloader")
    
    # Options
    data_opt = {    "neg_num": data_cfg.getint("neg_num"),
                    "batch_size": data_cfg.getint("batch_size"),
                    "num_workers": data_cfg.getint("num_workers")
                }
    
    dl_opt = {  "batch_sampler": None,          "batch_size":data_cfg.getint("batch_size"),
                "collate_fn":collate_tuples,    "pin_memory":True,
                "num_workers":data_cfg.getint("num_workers"), "shuffle":True, "drop_last":True}
    
    # transform
    transform = build_transforms(cfg)["test"]

    # dataset
    val_db = build_dataset(args, cfg, transform, mode='val')

    # loader
    val_dl = DataLoader(val_db, **dl_opt)
    
    return val_dl
    
    
def build_transforms(cfg):
    data_cfg    = cfg["dataloader"]
    aug_cfg     = cfg["augmentaion"]
    
    tfs = {}
    
    # test
    tfs["test"] = ImagesTransform(max_size=data_cfg.getint("max_size"),
                                  mean=IMAGENET_DEFAULT_MEAN, 
                                  std=IMAGENET_DEFAULT_STD)
    
    # train
    tf_post = create_transform( input_size = data_cfg.getint("max_size"),
                                is_training=True,
                                no_aug=True,
                                interpolation="bilinear")
    
    tfs["train"] = ImagesTransform(max_size=data_cfg.getint("max_size"),
                                   postprocessing=tf_post)
    
    # train augment
    tf_pre, tf_aug, tf_post = create_transform( input_size=data_cfg.getint("max_size"),
                                                is_training=True,
                                                auto_augment=aug_cfg.get("auto_augment"),
                                                interpolation="random", 
                                                re_prob=0.25,
                                                re_mode="pixel",
                                                re_count=2,
                                                re_num_splits=0,
                                                separate=True)
    
    tfs["train_aug"] = ImagesTransform(max_size=data_cfg.getint("max_size"),
                                       preprocessing=tf_pre,
                                       augmentation=tf_aug,
                                       postprocessing=tf_post)
    
    return tfs