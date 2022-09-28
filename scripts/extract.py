# General
import argparse
from os import makedirs, path
import numpy as np
import torch 

# Image Retrieval
from image_retrieval.feature_extractor import  FeatureExtractor
from image_retrieval.configuration import DEFAULTS as DEFAULT_CONFIG
from image_retrieval.utils.configurations  import make_config

from image_retrieval.utils.logging import setup_logger
from image_retrieval.datasets.generic import ImagesFromList


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


def main(args):
    
    # set device
    device_id, device = args.local_rank, torch.device(args.local_rank)
    torch.cuda.set_device(device_id)

    # set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # load cfguration
    cfg = make_config(args.config, defauls=DEFAULT_CONFIG["default"])

    # logger
    logger = setup_logger(output=args.directory, name="retrieval")
    
    # dataset
    dataset   = ImagesFromList(images_path="",
                               split="database",
                               max_size=1024)
    
    # extractor
    feature_extractor = FeatureExtractor(args=args, cfg=cfg, dataset=dataset)
    feature_extractor.extract()

if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())