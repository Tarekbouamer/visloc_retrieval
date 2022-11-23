# General
import argparse

from os import makedirs, path
import numpy as np
import torch 

import retrieval
from retrieval.configuration            import DEFAULTS as DEFAULT_CONFIG
from retrieval.utils.configurations     import make_config
from retrieval.tools.events            import EventWriter

from retrieval.tools.dataloader        import build_train_dataloader

import retrieval.datasets as data

from retrieval                  import  build_evaluator, create_model
from retrieval.utils.logging    import  setup_logger
from retrieval.utils.io         import  csv_float


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
    
    parser.add_argument("--scales", type=csv_float,
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
    
    args.directory = 'results'

    #
    logger = setup_logger(output=args.directory , name="retrieval", suffix=args.model)

    # load cfg file 
    cfg = make_config(args.config, defauls=DEFAULT_CONFIG["default"])
    
    # train data
    train_dl = build_train_dataloader(args=args, cfg=cfg)
    
    # evaluation meta 
    meta = {}
    meta['train_dl']= train_dl

    # create model
    model = create_model(args.model, cfg, pretrained=True)
        
    # evaluate
    evaluator = build_evaluator(args, cfg, model, None, None, **meta)
    evaluator.evaluate(scales=args.scales)

    #
    logger.info(f"extraction scales {args.scales}")

if __name__ == '__main__':

    args = make_parser()

    main(args)