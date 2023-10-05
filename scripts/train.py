import argparse
from os import makedirs, path

import numpy as np
import torch
from core.config import load_cfg
from core.logging import init_loguru

from retrieval.tools import ImageRetrievalTrainer


def make_parser():

    # ArgumentParser
    parser = argparse.ArgumentParser(
        description='VISLOC:: Image Retrieval Training')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--directory',  metavar='EXPORT_DIR',
                        help='experiments folder')
    parser.add_argument('--data',       metavar='EXPORT_DIR',
                        help='dataset folder')
    parser.add_argument("--config",     metavar="FILE",         type=str, help="cfg file",
                        default='retrieval/configuration/default.yaml')
    parser.add_argument("--eval",       action="store_true",
                        help="do evaluation")
    parser.add_argument('--resume',     action="store_true",
                        help='resume from experiment folder')

    return parser.parse_args()


def make_dir(cfg, directory):

    extension = "{}".format(cfg.dataloader.dataset)
    extension += "_{}".format(cfg.body.arch)
    extension += "_{}_m{:.2f}".format(cfg.loss.type,
                                      cfg.loss.margin)

    extension += "_{}_lr{:.1e}_wd{:.1e}".format(cfg.optimizer.type,
                                                cfg.optimizer.lr,
                                                cfg.optimizer.weight_decay)

    extension += "_nnum{}".format(cfg.dataloader.neg_num)

    extension += "_bsize{}_imsize{}".format(cfg.dataloader.batch_size,
                                            cfg.dataloader.max_size)

    # make directory
    directory = path.join(directory, extension)

    if not path.exists(directory):
        makedirs(directory)

    return directory


def main(args):

    # initialize device
    device_id, device = args.local_rank, torch.device(args.local_rank)  # noqa: F841
    torch.cuda.set_device(device_id)

    # random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # load cfg file
    cfg = load_cfg(args.config)

    # create experiment folder
    args.directory = make_dir(cfg, args.directory)

    # init retrieval logger
    logger = init_loguru(
        name="Retrieval", log_file=args.directory, file_name="training")

    # train
    trainer = ImageRetrievalTrainer(args=args, cfg=cfg)
    trainer.train()

    #
    logger.info("Done!")


if __name__ == '__main__':

    args = make_parser()

    main(args)
