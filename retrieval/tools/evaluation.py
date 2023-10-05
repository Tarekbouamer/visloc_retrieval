import os
from collections import OrderedDict

import torch
from loguru import logger

import retrieval.test.asmk as eval_asmk
from retrieval.test import build_paris_oxford_dataset, test_asmk, test_global_descriptor

from .dataloader import build_sample_dataloader


class DatasetEvaluator:

    def __init__(self, cfg, extractor, writer=None):

        # cfg
        self.cfg = cfg

        # feature extractor
        self.extractor = extractor

        # writer
        self.writer = writer

    def write_metrics(self, metrics, datatset, step=0, scale=1):
        """ write metrics to tensorboard """
        if self.writer is None:
            return

        # push metrics to history
        if len(metrics) > 0:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    self.writer.put(datatset + "/" + k, v * scale)
                else:
                    self.writer.put(datatset + "/" + k,
                                    torch.as_tensor(v * scale))

        # write to board
        self.writer.write(step)

    def evaluate(self):
        """ evaluate """
        raise NotImplementedError


class GlobalEvaluator(DatasetEvaluator):
    def __init__(self, cfg, extractor, writer=None, **kwargs):
        super().__init__(cfg, extractor, writer)

        # logger
        logger.info(f"set evaluator on ({self.cfg.test.mode}) mode")

    def evaluate(self, dataset, query_dl, database_dl, ground_truth=None):
        """ evaluate """

        # result
        results = OrderedDict()

        # writer on test mode
        if self.writer is not None:
            self.writer.test()

        # test
        metrics = test_global_descriptor(dataset,
                                         query_dl,
                                         database_dl,
                                         self.extractor,
                                         ground_truth=ground_truth)

        # write
        self.write_metrics(metrics, dataset, scale=100)

        # map
        results[dataset] = metrics["map"]

        return results


class ASMKEvaluator(DatasetEvaluator):

    def __init__(self, args, cfg, extractor, writer=None, **kwargs):
        super().__init__(args, cfg, extractor, writer)

        # train dataset
        self.train_dl = kwargs.pop('train_dl', None)

        # number of sampled image
        self.num_samples = cfg.test.num_samples

        #
        logger.info(f"set evaluator on ({self.cfg.test.mode}) mode")

    def build_codebook(self, args):

        # eval mode
        self.extractor.eval()

        logger.info('init asmk')
        asmk, params = eval_asmk.asmk_init()

        # train codebook
        save_path = os.path.join(
            args.directory, self.cfg.dataloader.dataset + "_codebook.pkl")

        # sample loader
        sample_dl = build_sample_dataloader(self.train_dl,
                                            num_samples=self.num_samples,
                                            cfg=self.cfg)

        logger.info(f'train codebook {len(sample_dl)} :   {save_path}')

        # train_codebook
        self.asmk = eval_asmk.train_codebook(self.cfg, sample_dl, self.extractor, asmk,
                                             scales=args.scales,
                                             save_path=save_path)

        return asmk

    def build_test_dataset(self, data_path, dataset):

        query_dl, database_dl, ground_truth = None, None, None

        if dataset in ['roxford5k', 'rparis6k', "val_eccv20"]:
            query_dl, database_dl, ground_truth = build_paris_oxford_dataset(
                data_path, dataset, self.cfg)
        else:
            raise KeyError

        return query_dl, database_dl, ground_truth

    def evaluate(self, args):

        # eval mode
        self.extractor.eval()

        # writer on test mode
        if self.writer is not None:
            self.writer.test()

        # check data path
        if not os.path.exists(args.data):
            logger.error("path not found: {args.data}")

        # build and save asmk codebook
        self.build_codebook(args.scales)

        # result dictionary
        results = OrderedDict()

        # eval all test_datasets
        for dataset in self.cfg.test.datasets:

            # build dataset
            query_dl, database_dl, ground_truth = self.build_test_dataset(
                args.data, dataset)

            # test
            metrics = test_asmk(dataset, query_dl, database_dl,
                                self.extractor,
                                args.scales, ground_truth, self.asmk)

            # write
            self.write_metrics(metrics, dataset, scale=100)

            # map
            results[dataset] = metrics["map"]

        return results


def build_evaluator(cfg, extractor, writer, **meta):

    if cfg.test.mode == 'global':
        return GlobalEvaluator(cfg, extractor, writer, **meta)
    elif cfg.test.mode == 'asmk':
        return ASMKEvaluator(cfg, extractor, writer, **meta)
    else:
        raise KeyError(f"mode {cfg.test.mode} not supported, \
                        available modes: ['global', 'asmk']")
