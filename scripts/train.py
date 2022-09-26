# General
import argparse
from os import makedirs, path
import numpy as np
import torch 

# Image Retrieval
from image_retrieval.tools import  ImageRetrievalTrainer
from image_retrieval.configuration import DEFAULTS as DEFAULT_CONFIG
from image_retrieval.utils.configurations  import make_config

from image_retrieval.utils.logging import setup_logger


def make_parser():
    
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Image Retrieval Training')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to cfguration file",
                        default='./cirtorch/cfguration/defaults/global_cfg.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run")

    parser.add_argument('--resume', action="store_true",
                        help='name of the latest checkpoint (default: None)')

    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")

    args = parser.parse_args()

    # for arg, value in sorted(vars(args).items()):
    #     logger.info("%s: %r", arg, value)

    return parser


def make_dir(cfg, directory):
    # Create export dir name if it doesnt exist in your experiment folder
    extension = "{}".format(cfg["dataloader"].get("dataset"))
    extension += "_{}".format(cfg["body"].get("arch"))

    extension += "_{}_m{:.2f}".format(cfg["global"].get("loss"),
                                      cfg["global"].getfloat("loss_margin"))

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

    directory = path.join(directory, extension)

    if not path.exists(directory):
        makedirs(directory)
        
    return directory

def main(args):
    
    # initialize multi-processing
    device_id, device = args.local_rank, torch.device(args.local_rank)

    # set device
    torch.cuda.set_device(device_id)

    # set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # load cfguration
    cfg = make_config(args.config, defauls=DEFAULT_CONFIG["default"])

    # experiment path
    args.directory = make_dir(cfg, args.directory)
    
    # logger
    logger = setup_logger(output=args.directory, name="retrieval")
    
    trainer = ImageRetrievalTrainer(args=args, cfg=cfg)
    trainer.train()

if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())