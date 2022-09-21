import numpy as np
import logging

from  torch.utils.data import DataLoader, SubsetRandomSampler, default_collate
from timm.data import create_transform

# core 
from core.utils.options     import test_datasets_names
from core.utils.download    import download_train, download_test

# image_retrieval
from image_retrieval.datasets.tuples            import TuplesDataset
from image_retrieval.datasets.generic import ImagesFromList, ImagesTransform, INPUTS


from image_retrieval.datasets.misc              import iss_collate_fn, collate_fn, collate_tuples

# logger
import logging
logger = logging.getLogger("retrieval")  

def build_sample_dataloader(cfg, images):

    #
    num_samples = cfg["global"].getint("num_samples")
    
    if num_samples > len(images):
        num_samples = len(images)
    
    if logger:
        logger.debug(f"sample dataset:  {num_samples}")

    #
    sampler     = SubsetRandomSampler(np.random.choice(len(images), num_samples, replace=False))
    
    db_tf       = ImagesTransform(max_size=cfg["dataloader"].getint("max_size"), 
                                  mean=cfg["augmentaion"].getstruct("mean"), 
                                  std=cfg["augmentaion"].getstruct("std"))
             
    sample_dl   = DataLoader(dataset=ImagesFromList("", images, transform=db_tf),
                            num_workers=cfg["dataloader"].getint("num_workers"), 
                            pin_memory=True,
                            sampler=sampler)
    
    return sample_dl


def make_dataloader(args, config, rank=None, world_size=None, non_augment=False, logger=None, **varargs):
  
  
    data_cfg    = config["dataloader"]
    aug_cfg     = config["augmentaion"]
    test_cfg    = config["test"]

    # Manually check if there are unknown test datasets
    for dataset in test_cfg.getstruct("datasets"):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

    # Check if test datasets are available, download it if  not !
    name = data_cfg.get("dataset")
    
    if name.startswith('retrieval-SfM'):
        download_train(args.data)
        download_test(args.data)

    # Data Loader
    logger.debug("Creating dataloaders for dataset in %s", args.data)
    
    # options 
    
    data_opt = {    "neg_num": data_cfg.getint("neg_num"),
                    "batch_size": data_cfg.getint("batch_size"),
                    "num_workers": data_cfg.getint("num_workers")
                }
    
    dl_opt = {  "batch_sampler": None,          "batch_size":data_cfg.getint("batch_size"),
                "collate_fn":iss_collate_fn,    "pin_memory":True,
                "num_workers":data_cfg.getint("num_workers"), "shuffle":True, "drop_last":True}
    
    
    # Data Augmentation and Transform
    tf_pre, tf_aug, tf_post = create_transform(
                                                input_size = data_cfg.getint("max_size"),
                                                is_training=True,
                                                no_aug=aug_cfg.getboolean("no_aug"),
                                                scale=aug_cfg.getstruct("scale"),
                                                ratio=aug_cfg.getstruct("ratio"),
                                                hflip=aug_cfg.getfloat("hflip"),
                                                vflip=aug_cfg.getfloat("vflip"),
                                                color_jitter=aug_cfg.getfloat("color_jitter"),
                                                auto_augment=aug_cfg.get("auto_augment"),
                                                interpolation=aug_cfg.get("interpolation"),
                                                mean=aug_cfg.getstruct("mean"),
                                                std=aug_cfg.getstruct("std"),
                                                
                                                crop_pct=None,
                                                tf_preprocessing=False,
                                                re_prob=aug_cfg.getfloat("re_prob"),
                                                re_mode=aug_cfg.get("re_mode"),
                                                re_count=aug_cfg.getint("re_count"),
                                                re_num_splits=0,
                                                
                                                separate=aug_cfg.getboolean("separate"),
    )
    
    # Transforms
    tf_aug = ImagesTransform(max_size=data_cfg.getint("max_size"),
                               preprocessing=tf_pre,
                               augmentation=tf_aug,
                               postprocessing=tf_post,
                               is_train=True)
    
    tf_non_aug = tf_aug
    tf_non_aug.is_train = False

    # Train
    train_db = TuplesDataset(root_dir=args.data,
                             name=data_cfg.get("dataset"),
                             mode='train',
                             query_size=data_cfg.getint("query_size"),
                             pool_size=data_cfg.getint("pool_size"),
                             transform=tf_aug,
                             **data_opt)

    train_dl = DataLoader(train_db, **dl_opt)


    # Train non augment            logger = logging.getLogger(__name__)

    train_naug_db = train_db
    train_naug_db.transform = tf_non_aug

    train_naug_dl = DataLoader(train_naug_db, **dl_opt)
    
    
    # Val 
    val_db = TuplesDataset(root_dir=args.data,
                           name=data_cfg.get("dataset"),
                           mode='val',
                           query_size=float('inf'),
                           pool_size=float('inf'),
                           transform=tf_non_aug,
                           **data_opt)

    val_dl = DataLoader(val_db, **dl_opt)    
    
    if non_augment:
        return train_naug_dl, val_dl
    
    return train_dl, val_dl


def build_train_dataloader(args, cfg):
    data_cfg    = cfg["dataloader"]
    aug_cfg     = cfg["augmentaion"]
    test_cfg    = cfg["test"]
    
    logger.info("build dataloader")
    
    # Options
    data_opt = {    "neg_num": data_cfg.getint("neg_num"),
                    "batch_size": data_cfg.getint("batch_size"),
                    "num_workers": data_cfg.getint("num_workers")
                }
    
    dl_opt = {  "batch_sampler": None,          "batch_size":data_cfg.getint("batch_size"),
                "collate_fn":collate_tuples,    "pin_memory":True,
                "num_workers":data_cfg.getint("num_workers"), "shuffle":True, "drop_last":True}
    
    
    # augmentation transforms
    tf_pre, tf_aug, tf_post = create_transform( input_size = data_cfg.getint("max_size"),
                                                is_training=True,
                                                hflip=0.3,
                                                color_jitter=0.4,
                                                auto_augment=aug_cfg.get("auto_augment"),
                                                interpolation="bilinear",         
                                                re_prob=0.25,
                                                re_mode="pixel",
                                                re_count=2,
                                                re_num_splits=0,
                                                separate=True)
    
    # transforms
    transform = ImagesTransform(max_size=data_cfg.getint("max_size"),
                               preprocessing=tf_pre,
                               augmentation=tf_aug,
                               postprocessing=tf_post,
                               is_train=False)
    
    # dataset
    train_db = TuplesDataset(root_dir=args.data,
                             name=data_cfg.get("dataset"),
                             mode='train',
                             query_size=data_cfg.getint("query_size"),
                             pool_size=data_cfg.getint("pool_size"),
                             transform=transform,
                             **data_opt)
    
    # loader
    train_dl = DataLoader(train_db, **dl_opt)
    
    return train_dl
    
