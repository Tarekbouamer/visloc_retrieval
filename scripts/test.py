# General
import argparse

from core.config import load_cfg
from core.logging import init_loguru

from retrieval.extractors import GlobalExtractor
from retrieval.models.misc import create_retrieval
from retrieval.test import build_paris_oxford_dataset
from retrieval.test.global_desc import test_global
from retrieval.utils.io import csv_float


def make_test_global_parser():

    parser = argparse.ArgumentParser(
        description='retrieval global testing parser')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to images folder')

    parser.add_argument("--model", metavar="MODEL", type=str, default='resnet101_gem_2048',
                        help="model name from models factory")

    parser.add_argument("--config", metavar="CFG", type=str, default='retrieval/configuration/default.yaml',
                        help="path to config file ini")

    parser.add_argument("--scales", type=csv_float, default=[1.0],
                        help="images scales ")

    parser.add_argument("--save_path", metavar="IMPORT_DIR", type=str,  default='.',
                        help="path where features are saved as h5py format ")

    parser.add_argument("--name", metavar="FILE", type=str, default='db',
                        help="name of stored file ")

    return parser.parse_args()


def run_test_global():

    # args
    args = make_test_global_parser()

    # init logger
    logger = init_loguru(
        name="Retrieval", log_file='.results', file_name="testing")

    # load cfg
    cfg = load_cfg(args.config)

    # create model
    model = create_retrieval(args.model, cfg, pretrained=True)

    # extractor
    extractor = GlobalExtractor(model=model, cfg=cfg)
    extractor.eval()

    for dataset in cfg.test.datasets:
        # build dataset
        query_dl, db_dl, gt = build_paris_oxford_dataset(args.data,
                                                         dataset,
                                                         cfg)
        # test
        test_global(dataset,
                    query_dl,
                    db_dl,
                    extractor,
                    ground_truth=gt)

    #
    logger.success("done")


if __name__ == '__main__':

    # run test
    run_test_global()
