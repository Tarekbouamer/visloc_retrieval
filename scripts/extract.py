# General
import argparse
from os import path

import retrieval
import retrieval.datasets as data
from core.logging import init_loguru

MODES = ['global', 'locals']


def make_parser():
    
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Image Retrieval Training')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to images folder')

    parser.add_argument("--model", metavar="MODEL", type=str, 
                        help="model name from models factory",
                        default='resnet101_gem_2048')

    parser.add_argument("--mode", metavar="MODE", type=str, 
                        help=f"type of features {MODES}",
                        default='global')
   
    parser.add_argument("--max_size", type=int, 
                        help="max image size",
                        default='100')
    
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
    
    # init retrieval logger
    logger = init_loguru(name="Retrieval", log_file=".", file_name="testing")    
    
    args.save_path = path.join(args.save_path, args.name + '.h5')

    #
    logger.info(f"loading images from {args.data}")
    
    # dataset
    transform   = data.ImagesTransform(max_size=args.max_size)
    dataset     = data.ImagesFromList(args.data, transform=transform)

    logger.info(f"#images {len(dataset)} img_size {args.max_size}")

    # extractor
    feature_extractor = retrieval.GlobalExtractor(args.model)
    
    # run
    if args.mode == 'global':
        pred = feature_extractor.extract_global(dataset, scales=args.scales, save_path=args.save_path)
    elif args.mode == 'local':
        pred = feature_extractor.extract_locals(dataset, scales=args.scales, save_path=args.save_path)
    else:
        raise KeyError()
    
    # summary
    logger.info(f"feature extractor {args.model}  dim {pred['features'].shape[-1]}")
    logger.info(f"extraction scales {args.scales}")
    logger.info(f"feature saved {args.save_path}")

if __name__ == '__main__':

    args = make_parser()

    main(args)