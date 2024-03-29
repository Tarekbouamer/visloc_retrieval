import numpy as np
from loguru import logger
from timm.data import create_transform
from torch.utils.data import DataLoader, SubsetRandomSampler

from retrieval.datasets import (
    GoogleLandmarkDataset,
    ImagesFromList,
    ImagesTransform,
    SatDataset,
    SfMDataset,
    collate_tuples,
)

DATASETS = ["retrieval-SfM-120k", "gl18", "gl20", "SAT"]


def build_dataset(args, cfg, transform, mode='train'):
    """ build dataset  """

    name = cfg.data.dataset

    if name == "retrieval-SfM-120k":
        return SfMDataset(data_path=args.data,
                          name=name,
                          cfg=cfg,
                          mode=mode,
                          transform=transform)

    # sfm
    if name in ["gl18", "gl20"]:
        return GoogleLandmarkDataset(data_path=args.data,
                                     name=name,
                                     cfg=cfg,
                                     mode=mode,
                                     transform=transform)
    elif name == "SAT":
        return SatDataset(root_dir=args.data,
                          name=name,
                          mode=mode,
                          query_size=cfg.data.query_size,
                          pool_size=cfg.data.pool_size,
                          transform=transform,
                          neg_num=cfg.data.neg_num)
    else:
        raise ValueError(f"dataset {name} not supported, \
                         available datasets: {DATASETS}")


def build_sample_dataloader(train_dl, num_samples=None, cfg=None):
    """ build sample dataloader from train dataloader """

    #
    images = train_dl.dataset.images

    #
    if num_samples is None:
        num_samples = cfg.pca.num_samples

    if num_samples > len(images):
        num_samples = len(images)

    logger.debug(f"Sample dataset:  {num_samples}")

    #
    sampler = SubsetRandomSampler(np.random.choice(
        len(images), num_samples, replace=False))

    # transform
    transform = build_transforms(cfg)["test"]

    # loader
    sample_dl = DataLoader(dataset=ImagesFromList("", images, transform=transform),
                           num_workers=1,
                           pin_memory=True,
                           sampler=sampler)

    return sample_dl


def build_train_dataloader(args, cfg):
    """ build train dataloader """

    # logger
    logger.info("Build train dataloader")

    # Options
    dl_opt = {"batch_sampler": None,          "batch_size": cfg.data.batch_size,
              "collate_fn": collate_tuples,    "pin_memory": True,
              "num_workers": cfg.data.num_workers, "shuffle": True, "drop_last": True}

    # transforms
    # transform = build_transforms(cfg)["train_aug"]
    transform = build_transforms(cfg)["train"]

    # dataset
    train_db = build_dataset(args, cfg, transform, mode='train')

    # dl
    return DataLoader(train_db, **dl_opt)


def build_val_dataloader(args, cfg):
    """ build val dataloader """

    logger.info("Build val dataloader")

    # Options
    {"neg_num": cfg.data.neg_num,
     "batch_size": cfg.data.batch_size,
     "num_workers": cfg.data.num_workers
     }

    dl_opt = {"batch_sampler": None,          "batch_size": cfg.data.batch_size,
              "collate_fn": collate_tuples,    "pin_memory": True,
              "num_workers": cfg.data.num_workers, "shuffle": True, "drop_last": True}

    # transform
    transform = build_transforms(cfg)["test"]

    # dataset
    val_db = build_dataset(args, cfg, transform, mode='val')

    # dl
    return DataLoader(val_db, **dl_opt)


def build_transforms(cfg):
    """ build transforms """

    tfs = {}

    # test
    tfs["test"] = ImagesTransform(max_size=cfg.data.max_size)

    tfs["train"] = ImagesTransform(max_size=cfg.data.max_size)

    # train augment
    tf_pre, tf_aug, _ = create_transform(input_size=cfg.data.max_size,
                                         is_training=True,
                                         auto_augment=cfg.augmentaion.auto_augment,
                                         interpolation="random",
                                         re_prob=0.25,
                                         re_mode="pixel",
                                         re_count=2,
                                         re_num_splits=0,
                                         separate=True)

    tfs["train_aug"] = ImagesTransform(max_size=cfg.data.max_size,
                                       preprocessing=tf_pre,
                                       augmentation=tf_aug)

    return tfs
