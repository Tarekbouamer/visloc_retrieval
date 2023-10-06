import os
import shutil

import torch
from core.logging import _log_api_usage
from loguru import logger
from timm.utils import ModelEmaV2

from retrieval.extractors.global_extractor import GlobalExtractor
from retrieval.models import create_retrieval
from retrieval.test.paris_oxford_benchmark import build_paris_oxford_dataset
from retrieval.tools.dataloader import (
    build_sample_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from retrieval.tools.evaluation import build_evaluator
from retrieval.tools.events import EventWriter
from retrieval.tools.loss import build_loss

#
from retrieval.tools.optimizer import build_lr_scheduler, build_optimizer
from retrieval.utils.snapshot import resume_from_snapshot, save_snapshot

from .base import TrainerBase


class ImageRetrievalTrainer(TrainerBase):
    def __init__(self, args, cfg):
        super().__init__(cfg)

        # args
        self.args = args

        # writer
        self.writer = self.build_writers(args, cfg)

        # build model
        self._model = self.build_model(cfg.body.name,
                                       cfg,
                                       pretrained=cfg.body.pretrained)

        # train and val dataloader
        self.train_dl = self.build_train_loader(args, cfg)
        self.val_dl = self.build_val_loader(args, cfg)

        # ema
        self._ema_model = self.build_ema_model(args, cfg)
        print(self._ema_model)

        # scheduler
        self.optimizer = self.build_optimizer(cfg, self._model)

        # loss
        self.loss = self.build_training_loss(cfg)

        # scheduler
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # resume
        if args.resume:
            self.resume_or_load()
        else:
            self.init_model()

        # evaluator
        if args.eval:
            self.evaluator = self.build_evaluator(args, cfg)
        else:
            self.evaluator = None

        logger.info("init trainer")

        # evaluate 0 epoch
        # if args.eval:
        #     self.test()

    # def get_dataset(self):
    #     if self.train_dl is not None:
    #         return self.train_dl.dataset.images

    def resume_or_load(self, resume=True):
        """
            resume training from checkpoint
        """

        # model
        snapshot_last_path = os.path.join(
            self.args.directory, "last_model.pth.tar")
        logger.info(f"resume load model {snapshot_last_path}")

        # load
        snapshot_last = resume_from_snapshot(
            self._model, snapshot_last_path, ["body", "head"])

        # optimizer
        self.optimizer.load_state_dict(
            snapshot_last["state_dict"]["optimizer"])

        self._start_epoch = self._epoch = snapshot_last["training_meta"]["epoch"] + 1
        self._best_score = snapshot_last["training_meta"]["best_score"]
        self._global_step = snapshot_last["training_meta"]["global_step"]

        # set metrics
        self.writer.set(snapshot_last["state_dict"])

        del snapshot_last

    def before_epoch(self):
        """ before epoch 
            - refresh train data
            - set model to train
            - zero grad
        """
        logger.info(f"learning rates {self.scheduler.get_lr()}")
        for it, lr_i in enumerate(self.scheduler.get_lr()):
            self.writer.add_scalar(f'lr/{it}', lr_i, self._epoch)

        # refresh train data
        self.refresh_train_data()

        # set model  to train
        if not self._model.training:
            self._model.train()

        # zero grad
        self.optimizer.zero_grad()

        # set writer to train
        self.writer.train()

    def after_epoch(self):

        snapshot_last = os.path.join(
            self.args.directory, "last_model.pth.tar")

        logger.info(f"save snapshot:    {snapshot_last}")

        # get metrics
        metrics = self.writer.get()
        meters_out = {k: v.state_dict() for k, v in metrics.items()}

        # save model last
        save_snapshot(snapshot_last, self.cfg, self._epoch, self._last_score,
                      self._best_score,
                      self._global_step,
                      body=self._model.body.state_dict(),
                      head=self._model.head.state_dict(),
                      optimizer=self.optimizer.state_dict(),
                      **meters_out
                      )

        # save model ema
        if self._ema_model is not None:
            snapshot_ema = os.path.join(
                self.args.directory, "ema_model.pth.tar")

            logger.info(f"save ema snapshot:    {snapshot_ema}")

            save_snapshot(snapshot_ema, self.cfg, self._epoch, self._last_score, self._best_score, self._global_step,
                          body=self._ema_model.module.body.state_dict(),
                          head=self._ema_model.module.head.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          **meters_out
                          )

        # update learning rate
        self.scheduler.step()

    def val_epoch(self):
        if (self._epoch % self.cfg["general"].val_interval) != 0:
            return

        logger.info(f"val epoch {self._epoch}")
        super().val_epoch()

    def test_epoch(self):

        if (self._epoch % self.cfg["general"].test_interval) != 0:
            return

        logger.info(f"test epoch {self._epoch}")

        # test
        if self.evaluator is not None:
            self._last_score = self.test()

        # load snapshot
        if self._ema_model is not None:
            snapshot_path = os.path.join(
                self.args.directory, "ema_model.pth.tar")
            snapshot = torch.load(snapshot_path, map_location="cpu")
        else:
            snapshot_path = os.path.join(
                self.args.directory, "last_model.pth.tar")
            snapshot = torch.load(snapshot_path, map_location="cpu")

        # update score
        snapshot["training_meta"]["last_score"] = self._last_score
        torch.save(snapshot, snapshot_path)

        # save best
        if (self._best_score is None) or \
                (self._last_score >= self._best_score):

            best_snapshot_path = os.path.join(
                self.args.directory, "best_model.pth.tar")

            self._best_score = self._last_score
            shutil.copy(snapshot_path, best_snapshot_path)
            logger.info(f"save best snapshot:   {best_snapshot_path}")

    def init_model(self):
        if hasattr(self._model, '_init_model'):
            #
            logger.info("init model layers")
            sample_dl = build_sample_dataloader(self.train_dl, cfg=self.cfg)

            #
            self._model._init_model(
                self.args, self.cfg,  self._model, sample_dl)
            logger.info("init model done")

    def before_train(self, ):
        """ before train 
            - get sample data
            - init model
        """
        logger.info("initilize model layers, using pca")

        # sample data
        sample_dl = build_sample_dataloader(self.train_dl, cfg=self.cfg)

        # init model
        self._model.init_model(sample_dl, save_path=self.args.directory)

    def train(self):
        """
          Run training.
        """
        logger.info("start training")
        super().train(self._start_epoch, self._max_epochs)

    def train_epoch(self):
        logger.info(f"train epoch {self._epoch}")
        super().train_epoch()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    def refresh_train_data(self):
        logger.info("refresh train dataset")
        stats = self.train_dl.dataset.create_epoch_tuples(
            self.cfg, self._model)

        if len(stats) > 0:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    self.writer.add_scalar(k, v, self._epoch)
                else:
                    self.writer.add_scalar(k, torch.as_tensor(v), self._epoch)

    def refresh_val_data(self):
        logger.info("refresh val dataset")
        stats = self.val_dl.dataset.create_epoch_tuples(self.cfg, self._model)

        if len(stats) > 0:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    self.writer.add_scalar(k, v, self._epoch)
                else:
                    self.writer.add_scalar(k, torch.as_tensor(v), self._epoch)

    def write_metrics(self, metrics, step, max_steps, global_step=None, prefix=""):
        """
        Args:
            metrics (dict): losses, batch and data time 
        """

        metrics_dict = {k: v.detach().cpu() for k, v in metrics.items()}

        # push metrics to history
        if len(metrics_dict) > 0:
            for k, v in metrics_dict.items():
                if isinstance(v, torch.Tensor):
                    self.writer.put(k, v)

        # write to board
        self.writer.write(global_step if global_step else step)

        # logger iteration
        if step % self.writer.print_freq == 0:
            self.writer.log(self._epoch, self._max_epochs,  step, max_steps)

    @classmethod
    def build_model(self, model_name, cfg, pretrained=False):
        """ build model """

        # build model
        model = create_retrieval(model_name, cfg=cfg, pretrained=pretrained)
        model.cuda()

        _log_api_usage("modeling.meta_arch." + model_name)

        logger.info(f"model:\n {model}")

        return model

    def build_ema_model(self, args, cfg):
        """ build ema model and initialize if resume """

        # build EmaV2 model
        ema_model = ModelEmaV2(self._model, decay=cfg.body.ema_decay)

        # resume
        # if args.resume:
        #     # model
        #     snapshot_ema_path = os.path.join(
        #         self.args.directory, "ema_model.pth.tar")
        #     logger.info(f"resume load ema model {snapshot_ema_path}")

        #     # load
        #     snapshot_last = resume_from_snapshot(
        #         ema_model.module, snapshot_ema_path, ["body", "head"])

        #     self._best_score = snapshot_last["training_meta"]["best_score"]

        #     del snapshot_last

        return ema_model

    @classmethod
    def build_optimizer(self, cfg, model):
        """
            Build optimizer for training  
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(self, cfg, optimizer):
        """
            Build learning scheduler 
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(self, args, cfg):
        """ build training dataloader """
        return build_train_dataloader(args, cfg)

    @classmethod
    def build_val_loader(self, args, cfg):
        """ build validation dataloader"""
        return build_val_dataloader(args, cfg)

    @classmethod
    def build_training_loss(self, cfg):
        """ build training loss """
        return build_loss(cfg)

    @classmethod
    def build_writers(self, args, cfg):
        """build training writer"""
        return EventWriter(args.directory, cfg["general"].log_interval)

    def build_evaluator(self, args, cfg):
        """ build evaluator """

        if self._ema_model is not None:
            model = self._ema_model.module
        else:
            model = self._model

        print(model)

        # extractor
        extractor = GlobalExtractor(cfg, model=model)
        extractor.eval()

        return build_evaluator(cfg, extractor)

    def test(self):
        """ run test """

        logger.info(f"evaluate epoch {self._epoch}")

        results_dict = {}

        for dataset in self.cfg.test.datasets:

            # build dataset
            query_dl, db_dl, gt = build_paris_oxford_dataset(self.args.data,
                                                             dataset,
                                                             self.cfg)
            # evaluate
            results_dict[dataset] = self.evaluator.evaluate(dataset,
                                                            query_dl,
                                                            db_dl,
                                                            gt)

        # overall score
        scores = [v for k, v in results_dict.items()]
        score = sum(scores)/len(scores)

        logger.info(f"score: {score}")

        #
        return score
