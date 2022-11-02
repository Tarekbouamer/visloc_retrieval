# General
import argparse
from os import makedirs, path
import numpy as np
import torch 

import retrieval
from retrieval.configuration            import DEFAULTS as DEFAULT_CONFIG
from retrieval.utils.configurations     import make_config
from retrieval.tools.events            import EventWriter

import retrieval.datasets as data
from retrieval import  build_evaluator, create_model
from  retrieval.utils.logging import  setup_logger


def make_parser():
    
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Image Retrieval Training')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to images folder')

    parser.add_argument("--model", metavar="MODEL", type=str, 
                        help="model name from models factory",
                        default='resnet101_gem_2048')
    
    parser.add_argument("--config", metavar="CFG", type=str, 
                        help="path to config file ini",
                        default='retrieval/configuration/defaults/test.ini')
    
    parser.add_argument("--scales",
                        help="images scales ",
                        default=[1.0])

    parser.add_argument("--save_path", metavar="IMPORT_DIR", type=str,
                        help="path where features are saved as h5py format ",
                        default='.')

    parser.add_argument("--name", metavar="FILE", type=str,
                        help="name of stored file ",
                        default='db')  
    # 
    args = parser.parse_args()

    return args


def main(args):
    
    writer = EventWriter(directory='results', print_freq=1)
    
    logger = setup_logger(output="results", name="retrieval", suffix=args.model)

    args.save_path = path.join(args.save_path, args.name + '.h5')
    
    # load cfg file 
    cfg = make_config(args.config, defauls=DEFAULT_CONFIG["default"])
    
    # model
    model = create_model(args.model, cfg, pretrained=True)
    model.to(torch.device("cuda"))
    
    # 
    evaluator = build_evaluator(args, cfg, model, None, None)
    evaluator.evaluate()



    logger.info(f"extraction scales {args.scales}")
    logger.info(f"feature saved {args.save_path}")

if __name__ == '__main__':

    args = make_parser()

    main(args)