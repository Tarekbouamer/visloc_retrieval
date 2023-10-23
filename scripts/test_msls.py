# General
import argparse
import os

import faiss
import numpy as np
from core.config import load_cfg
from core.logging import init_loguru
from core.parser.types import float_list

from retrieval.extractors import GlobalExtractor
from retrieval.models.misc import create_retrieval
from retrieval.test.msls_benchmark import build_msls_dataset


def make_msls_global_parser():

    parser = argparse.ArgumentParser(
        description='retrieval msls testing parser')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to images folder')

    parser.add_argument("--model", metavar="MODEL", type=str, default='resnet101_gem_2048',
                        help="model name from models factory")

    parser.add_argument("--config", metavar="CFG", type=str, default='retrieval/configuration/test.yaml',
                        help="path to config file ini")

    parser.add_argument("--scales", type=float_list, default=None,
                        help="images scales ")

    parser.add_argument("--save_path", metavar="IMPORT_DIR", type=str,  default='.',
                        help="path where features are saved as h5py format ")

    parser.add_argument("--name", metavar="FILE", type=str, default='db',
                        help="name of stored file ")

    args = parser.parse_args()

    # scales
    if args.scales is None:
        args.scales = [1.0]

    return args


def run_test_msls():

    # args
    args = make_msls_global_parser()

    # save path
    save_path = "./results/msls/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init logger
    logger = init_loguru(
        name="Retrieval", log_file=save_path, file_name="testing")

    # load cfg
    cfg = load_cfg(args.config)

    # create model
    model = create_retrieval(args.model, cfg, pretrained=True)

    # extractor
    extractor = GlobalExtractor(model=model, cfg=cfg)
    extractor.eval()

    # results path
    results_file = os.path.join(save_path, f"{args.model}.txt")
    if os.path.exists(results_file):
        os.remove(results_file)

    # write results
    with open(results_file, 'w') as f_writer:

        for dataset in cfg.test.datasets:

            # test dataset
            logger.info(f"test dataset {dataset}")

            # build dataset
            query_dl, db_dl, _ = build_msls_dataset(args.data, dataset, cfg)

            # extract query
            q_vecs = extractor.extract(query_dl)

            # extract database
            db_vecs = extractor.extract(db_dl)

            # match
            logger.info(
                f"Matching query and database vectors pool_size {db_vecs.shape[1]}")
            faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
            faiss_index.add(db_vecs)

            # search
            logger.info("Searching ...")
            n_values = [1, 5, 10, 20, 50, 100]
            num_k = max(n_values)
            _, predictions = faiss_index.search(q_vecs, num_k)

            predictions_new = []
            for _, pred in enumerate(predictions):
                _, idx = np.unique(np.floor(pred).astype(
                    np.int_), return_index=True)
                pred = pred[np.sort(idx)]
                pred = pred[:num_k]
                predictions_new.append(pred)

            # numpy
            predictions = np.array(predictions_new)
            db_image_list = np.array(db_dl.dataset.images)
            query_image_list = np.array(query_dl.dataset.images)

            #
            for q_idx, preds_idx in enumerate(predictions):
                #
                preds_full_path = db_image_list[preds_idx]
                query_full_path = query_image_list[q_idx]

                # get base name without extension
                query_full_path = os.path.splitext(query_full_path)[0]
                query_name = os.path.basename(query_full_path).split(".")[0]

                # get base name without extension
                preds_base_names = []
                for pred_path in preds_full_path:
                    preds_base_names.append(
                        os.path.basename(pred_path).split(".")[0])

                # write
                txt = f"{query_name} {' '.join(preds_base_names)}\n"
                f_writer.write(txt)
    
    # log
    logger.info(f"Results saved {results_file}")
    logger.success("Done")


if __name__ == '__main__':

    # run test
    run_test_msls()
