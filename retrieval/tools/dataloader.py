import numpy as np
from loguru import logger
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, SubsetRandomSampler

from retrieval.datasets import (
    ImagesFromList,
    ImagesTransform,
    SatDataset,
    TuplesDataset,
)
from retrieval.datasets.misc import collate_tuples

DATASETS = ["retrieval-SfM-120k", "gl18", "gl20", "SAT"]

def build_dataset(args, cfg, transform, mode='train'):

    

    # sfm
    if cfg.dataloader.dataset in ["retrieval-SfM-120k", "gl18", "gl20"]:

        train_db = TuplesDataset(root_dir=args.data,
                                 name=cfg.dataloader.dataset,
                                 mode=mode,
                                 query_size=cfg.dataloader.query_size,
                                 pool_size=cfg.dataloader.pool_size,
                                 transform=transform,
                                 neg_num=cfg.dataloader.neg_num)
    elif cfg.dataloader.dataset == "SAT":

        train_db = SatDataset(root_dir=args.data,
                              name=cfg.dataloader.dataset,
                              mode=mode,
                              query_size=cfg.dataloader.query_size,
                              pool_size=cfg.dataloader.pool_size,
                              transform=transform,
                              neg_num=cfg.dataloader.neg_num)
    else:
        raise ValueError(f"dataset {cfg.dataloader.get('dataset')} not supported, \
                         available datasets: {DATASETS}")

    return train_db


def build_sample_dataloader(train_dl, num_samples=None, cfg=None):

    #
    images = train_dl.dataset.images

    #
    if num_samples is None:
        num_samples = cfg.pca.num_samples

    if num_samples > len(images):
        num_samples = len(images)

    logger.debug(f"sample dataset:  {num_samples}")

    #
    sampler = SubsetRandomSampler(np.random.choice(
        len(images), num_samples, replace=False))

    # transform
    transform = build_transforms(cfg)["test"]

    # loader
    sample_dl = DataLoader(dataset=ImagesFromList("", images, transform=transform),
                           num_workers=cfg.dataloader.num_workers,
                           pin_memory=True,
                           sampler=sampler)

    return sample_dl


def build_train_dataloader(args, cfg):
    
    cfg.test

    logger.info("build train dataloader")

    # Options
    dl_opt = {"batch_sampler": None,          "batch_size": cfg.dataloader.batch_size,
              "collate_fn": collate_tuples,    "pin_memory": True,
              "num_workers": cfg.dataloader.num_workers, "shuffle": True, "drop_last": True}

    # transforms
    # transform = build_transforms(cfg)["train_aug"]
    transform = build_transforms(cfg)["train"]

    # dataset
    train_db = build_dataset(args, cfg, transform, mode='train')

    # loader
    train_dl = DataLoader(train_db, **dl_opt)

    return train_dl


def build_val_dataloader(args, cfg):
    

    logger.info("build val dataloader")

    # Options
    {"neg_num": cfg.dataloader.neg_num,
     "batch_size": cfg.dataloader.batch_size,
     "num_workers": cfg.dataloader.num_workers
     }

    dl_opt = {"batch_sampler": None,          "batch_size": cfg.dataloader.batch_size,
              "collate_fn": collate_tuples,    "pin_memory": True,
              "num_workers": cfg.dataloader.num_workers, "shuffle": True, "drop_last": True}

    # transform
    transform = build_transforms(cfg)["test"]

    # dataset
    val_db = build_dataset(args, cfg, transform, mode='val')

    # loader
    val_dl = DataLoader(val_db, **dl_opt)

    return val_dl


def build_transforms(cfg):
    
    aug_cfg = cfg.augmentaion

    tfs = {}

    # test
    tfs["test"] = ImagesTransform(max_size=cfg.dataloader.max_size,
                                  mean=IMAGENET_DEFAULT_MEAN,
                                  std=IMAGENET_DEFAULT_STD)

    # train
    tf_post = create_transform(input_size=cfg.dataloader.max_size,
                               is_training=True,
                               no_aug=True,
                               interpolation="bilinear")

    tfs["train"] = ImagesTransform(max_size=cfg.dataloader.max_size,
                                   postprocessing=tf_post)

    # train augment
    tf_pre, tf_aug, tf_post = create_transform(input_size=cfg.dataloader.max_size,
                                               is_training=True,
                                               auto_augment=aug_cfg.get(
                                                   "auto_augment"),
                                               interpolation="random",
                                               re_prob=0.25,
                                               re_mode="pixel",
                                               re_count=2,
                                               re_num_splits=0,
                                               separate=True)

    tfs["train_aug"] = ImagesTransform(max_size=cfg.dataloader.max_size,
                                       preprocessing=tf_pre,
                                       augmentation=tf_aug,
                                       postprocessing=tf_post)

    return tfs
