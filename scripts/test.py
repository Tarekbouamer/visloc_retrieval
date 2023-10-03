# General
import argparse

from core.config import load_cfg
from core.logging import init_loguru

from retrieval.models.misc import create_retrieval
from retrieval.tools.dataloader import build_train_dataloader
from retrieval.tools.evaluation import build_evaluator
from retrieval.utils.io import csv_float


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
                        default='retrieval/configuration/default.yaml')

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
    logger = init_loguru(
        name="retrieval", log_file=args.directory, file_name="testing")

    # load cfg file
    cfg = load_cfg(args.config)

    # train data
    train_dl = build_train_dataloader(args=args, cfg=cfg)

    # evaluation meta
    meta = {}
    meta['train_dl'] = train_dl

    # create model
    model = create_retrieval(args.model, cfg, pretrained=True)

    # evaluate
    evaluator = build_evaluator(args, cfg, model, None, None, **meta)
    evaluator.evaluate(scales=args.scales)

    #
    logger.info(f"extraction scales {args.scales}")


if __name__ == '__main__':

    args = make_parser()

    main(args)
