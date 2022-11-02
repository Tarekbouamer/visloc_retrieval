# General
import argparse
from os import makedirs, path
import numpy as np
import torch 

# Image Retrieval
from retrieval.tools                    import ImageRetrievalTrainer
from retrieval.configuration            import DEFAULTS as DEFAULT_CONFIG
from retrieval.utils.configurations     import make_config

from retrieval.utils.logging import setup_logger


def make_parser():
    
    # ArgumentParser
    parser = argparse.ArgumentParser(description='VISLOC:: Image Retrieval Training')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--directory',  metavar='EXPORT_DIR',   help='experiments folder')
    parser.add_argument('--data',       metavar='EXPORT_DIR',   help='dataset folder')
    parser.add_argument("--config",     metavar="FILE",         type=str, help="cfg file",      default='image_retrieval/configuration/defaults/default.ini')
    parser.add_argument("--eval",       action="store_true",    help="do evaluation")
    parser.add_argument('--resume',     action="store_true",    help='resume from experiment folder')

    return parser.parse_args()


def make_dir(cfg, directory):

    extension = "{}".format(cfg["dataloader"].get("dataset"))
    extension += "_{}".format(cfg["body"].get("arch"))
    extension += "_{}_m{:.2f}".format(cfg["global"].get("loss"), cfg["global"].getfloat("loss_margin"))

    if cfg["global"].getstruct("pooling"):
        extension += "_{}".format(cfg["global"].getstruct("pooling")["name"])
        
    if cfg["global"].get("attention"):
        extension += "_{}_encs{}_h{}".format(cfg["global"].get("attention"),
                                             cfg["global"].getfloat("num_encs"),
                                             cfg["global"].getfloat("num_heads"))

    extension += "_{}_lr{:.1e}_wd{:.1e}".format(cfg["optimizer"].get("type"),
                                                cfg["optimizer"].getfloat("lr"),
                                                cfg["optimizer"].getfloat("weight_decay"))

    extension += "_nnum{}".format(cfg["dataloader"].getint("neg_num"))
    
    extension += "_bsize{}_imsize{}".format(cfg["dataloader"].getint("batch_size"),
                                            cfg["dataloader"].getint("max_size"))

    # make directory
    directory = path.join(directory, extension)

    if not path.exists(directory):
        makedirs(directory)
        
    return directory


def main(args):
    
    # initialize device
    device_id, device = args.local_rank, torch.device(args.local_rank)
    torch.cuda.set_device(device_id)

    # random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # load cfg file 
    cfg = make_config(args.config, defauls=DEFAULT_CONFIG["default"])

    # create experiment folder
    args.directory = make_dir(cfg, args.directory)
    
    # init retrieval logger
    logger = setup_logger(output=args.directory, name="retrieval")
    
    # train
    trainer = ImageRetrievalTrainer(args=args, cfg=cfg)
    trainer.train()
    
    #
    logger.info("Done!")


if __name__ == '__main__':

    args = make_parser()

    main(args)