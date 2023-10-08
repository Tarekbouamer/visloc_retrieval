import shutil
import time
from os import path

import torch
from core.logging import _log_api_usage
from loguru import logger
from timm.utils import ModelEmaV2

from retrieval.extractors.global_extractor import GlobalExtractor
from retrieval.models import create_retrieval
from retrieval.test.paris_oxford_benchmark import build_paris_oxford_dataset

#
from retrieval.tools import (
    EventWriter,
    build_evaluator,
    build_loss,
    build_lr_scheduler,
    build_optimizer,
    build_sample_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from retrieval.utils.snapshot import resume_from_snapshot, save_snapshot

from .base import TrainerBase


class ImageRetrievalTrainer(TrainerBase):
    def __init__(self, args, cfg):
        super().__init__(cfg)

        # args
        self.args = args

        # workspace
        self._workspace = args.directory

        # writer
        self.writer = self.build_writers(self._workspace, cfg)

        # build model
        self._model = self.build_model(cfg.body.name,
                                       cfg,
                                       pretrained=cfg.body.pretrained)

        # build train and val dataloader
        self.train_dl = self.build_train_loader(args, cfg)
        self.val_dl = self.build_val_loader(args, cfg)

        # build ema model
        self._ema_model = self.build_ema_model(args, cfg)

        # build scheduler
        self.optimizer = self.build_optimizer(cfg, self._model)

        # build loss
        self.loss = self.build_training_loss(cfg)

        # build scheduler
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # resume
        if args.resume:
            self.resume_or_load()

        # build evaluator
        if args.eval:
            self.evaluator = self.build_evaluator(args, cfg)
        else:
            self.evaluator = None

        logger.info("init image retrieval trainer")

    def resume_or_load(self, resume=True):
        """ resume or load snapshot """

        # model
        snapshot_last_path = path.join(
            self._workspace, "last_model.pth.tar")
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

    def before_train(self, test=True):
        """ before train 
                1. set model to train
                2. init model layers
                3. test model (optional)
        """
        logger.info("initilize model layers, using pca")

        # sample data
        sample_dl = build_sample_dataloader(self.train_dl, cfg=self.cfg)

        # init model
        self._model.init_model(sample_dl, save_path=self._workspace)

        # test
        if test:
            self.test()

    def train_epoch(self):
        """ train epoch
                1. refresh train dataset
                2. set model to train & zero grad 
                3. train -> for each tuple
                    3.1 extract vectors
                    3.2 compute loss 
                    3.3 backward pass
                    3.4 optimize 
                4. write metrics
                5. save snapshot
                6. update learning rate
        """
        # learning rate
        logger.info(f"learning rates {self.scheduler.get_lr()}")

        for it, lr_i in enumerate(self.scheduler.get_lr()):
            self.writer.add_scalar(f'lr/{it}', lr_i, self._epoch)

        # refresh train data
        logger.info("refresh train dataset")
        self.refresh_train_data()

        # set model to train
        if not self._model.training:
            self._model.train()

        # zero grad
        self.optimizer.zero_grad()

        # set writer to train
        self.writer.train()
        logger.info(f"train epoch {self._epoch}")

        # timer
        data_time = time.time()

        # train ->
        for step, (tuples, target) in enumerate(self.train_dl):

            # global step
            self._global_step += 1

            # data_time
            data_time = torch.as_tensor(time.time() - data_time)

            #
            num_imgs = len(tuples[0])

            # batch time
            batch_time = time.time()

            # run -> for each tuple
            for tuple_i, target_i in zip(tuples, target):

                vecs = torch.zeros(num_imgs, self._model.dim).cuda()

                # extract vectors
                for n in range(len(tuple_i)):
                    data = {"image": tuple_i[n].cuda()}
                    pred = self._model(data, do_whitening=True)
                    vecs[n, :] = pred['features']

                # compute loss
                loss = self.loss(vecs, target_i.cuda())

                # backward
                loss.backward()

            # optimize
            self.optimizer.step()

            # ema update
            if self._ema_model is not None:
                self._ema_model.update(self._model)

            # zero grads
            self.optimizer.zero_grad()

            # batch time
            batch_time = torch.as_tensor(time.time() - batch_time)

            # metrics
            with torch.no_grad():
                if isinstance(loss, torch.Tensor):
                    metrics = {
                        "loss":         loss,
                        "total_loss":   loss,
                        "data_time":    data_time,
                        "batch_time":   batch_time}

                # write
                self.write_metrics(metrics, step, len(
                    self.train_dl), global_step=self._global_step)

                #
                self.writer.add_images(tuple_i, step)
                # self.writer.add_graph(self._model, tuple_i)

            #
            data_time = time.time()

        # write metrics
        metrics = self.writer.get()
        meters_out = {k: v.state_dict() for k, v in metrics.items()}

        # snapshot path
        snapshot_last = path.join(self._workspace, "last_model.pth.tar")
        logger.info(f"save snapshot: {snapshot_last}")

        # save model
        save_snapshot(snapshot_last, self.cfg, self._epoch, self._last_score,
                      self._best_score,
                      self._global_step,
                      model=self._model.state_dict(),
                      optimizer=self.optimizer.state_dict(),
                      **meters_out)

        # save model ema
        if self._ema_model is not None:

            # ema snapshot path
            snapshot_ema = path.join(self._workspace, "ema_model.pth.tar")
            logger.info(f"save ema snapshot: {snapshot_ema}")

            # save ema model
            save_snapshot(snapshot_ema, self.cfg, self._epoch, self._last_score,
                          self._best_score, self._global_step,
                          model=self._ema_model.module.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          **meters_out)

        # update learning rate
        self.scheduler.step()

    def val_epoch(self):
        """ validation epoch 
                1. refresh val data
                2. set model to eval
                3. val -> for each tuple
                    3.1 extract vectors
                    3.2 compute loss
                4. write metrics
        """

        # skip if not val interval
        if (self._epoch % self.cfg["general"].val_interval) != 0:
            return

        logger.info(f"val epoch {self._epoch}")

        # set to eval
        if self._model.training:
            self._model.eval()

        self.writer.eval()

        # refresh val data
        self.refresh_val_data()

        # timer
        data_time = time.time()

        # val ->
        for step, (tuples, target) in enumerate(self.val_dl):

            with torch.no_grad():

                # data_time
                data_time = torch.as_tensor(time.time() - data_time)
                num_imgs = len(tuples[0])

                # batch time
                batch_time = time.time()

                # run -> for each tuple
                for tuple_i, target_i in zip(tuples, target):

                    vecs = torch.zeros(num_imgs, self._model.dim).cuda()

                    # extract vectors
                    for n in range(len(tuple_i)):
                        data = {"image": tuple_i[n].cuda()}
                        pred = self._model(data, do_whitening=True)
                        vecs[n, :] = pred['features']

                    # compute loss
                    loss = self.loss(vecs, target_i.cuda())

                # batch time
                batch_time = torch.as_tensor(time.time() - batch_time)

                # metrics
                with torch.no_grad():
                    if isinstance(loss, torch.Tensor):
                        metrics = {
                            "loss":         loss,
                            "total_loss":   loss,
                            "data_time":    data_time,
                            "batch_time":   batch_time}

                    # write
                    self.write_metrics(metrics, step, len(self.val_dl))

                #
                data_time = time.time()

    def test_epoch(self):
        """ test epoch
                1. test the model using the evaluator
                2. add score to snapshot
        """

        # skip if not test interval
        if (self._epoch % self.cfg["general"].test_interval) != 0:
            return

        logger.info(f"test epoch {self._epoch}")

        # run test -> evaluate
        self._last_score = self.test()

        # load snapshot
        if self._ema_model is not None:
            snapshot_path = path.join(self._workspace, "ema_model.pth.tar")
            snapshot = torch.load(snapshot_path, map_location="cpu")
        else:
            snapshot_path = path.join(self._workspace, "last_model.pth.tar")
            snapshot = torch.load(snapshot_path, map_location="cpu")

        # update score
        snapshot["training_meta"]["last_score"] = self._last_score
        torch.save(snapshot, snapshot_path)

        # save best
        if (self._best_score is None) or (self._last_score >= self._best_score):
            # update best score
            self._best_score = self._last_score

            # save best
            best_snapshot_path = path.join(
                self._workspace, "best_model.pth.tar")

            shutil.copy(snapshot_path, best_snapshot_path)
            logger.info(f"save best snapshot: {best_snapshot_path}")

    def after_train(self):
        """ called after the training loop """
        pass

    def state_dict(self):
        """ state dict """
        # FIXME: not sure if this is the right way to do it
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """ load state dict """
        # FIXME: not sure if this is the right way to do it
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    def refresh_train_data(self):
        """ refresh train data """

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
        """ refresh val data """

        logger.info("refresh val dataset")

        stats = self.val_dl.dataset.create_epoch_tuples(
            self.cfg, self._model)

        if len(stats) > 0:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    self.writer.add_scalar(k, v, self._epoch)
                else:
                    self.writer.add_scalar(k, torch.as_tensor(v), self._epoch)

    def write_metrics(self, metrics, step, max_steps, global_step=None, prefix=""):
        """ write metrics to tensorboard """

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
        """ build model from factory"""

        # build model
        model = create_retrieval(model_name, cfg=cfg, pretrained=pretrained)
        model.cuda()

        # FIXME: this is a hack to log the model name
        _log_api_usage("modeling.meta_arch." + model_name)

        return model

    def build_ema_model(self, args, cfg):
        """ build ema model 
                1. build EmaV2 model
                2. load snapshot
        """

        # build EmaV2 model
        ema_model = ModelEmaV2(self._model, decay=cfg.body.ema_decay)

        # resume
        if args.resume:
            # model
            snapshot_ema_path = path.join(
                self._workspace, "ema_model.pth.tar")
            logger.info(f"resume load ema model {snapshot_ema_path}")

            # load
            snapshot_last = resume_from_snapshot(
                ema_model.module, snapshot_ema_path, ["body", "head"])

            self._best_score = snapshot_last["training_meta"]["best_score"]

            del snapshot_last

        return ema_model

    @classmethod
    def build_optimizer(self, cfg, model):
        """ build optimizer """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(self, cfg, optimizer):
        """ build learning rate scheduler """
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
    def build_writers(self, workspace, cfg):
        """build training writer"""
        return EventWriter(workspace, cfg["general"].log_interval)

    def build_evaluator(self, args, cfg):
        """ build evaluator 
                1. build extractor
                2. build evaluator
        """

        # model
        model = self._ema_model.module if self._ema_model is not None else self._model

        # extractor
        extractor = GlobalExtractor(cfg, model=model)
        extractor.eval()

        # build evaluator
        return build_evaluator(cfg, extractor)

    def test(self):
        """ test the model pipeline  """

        logger.info(f"test {self._epoch}")
        results_dict = {}

        # for each dataset
        for dataset in self.cfg.test.datasets:

            # build dataset
            query_dl, db_dl, gt = build_paris_oxford_dataset(
                self.args.data, dataset, self.cfg)

            # evaluate
            results_dict[dataset] = self.evaluator.evaluate(
                dataset, query_dl, db_dl, gt)

        # overall score
        scores = [v for k, v in results_dict.items()]
        score = sum(scores)/len(scores)

        logger.info(f"score: {score}")
        return score

    def train(self):
        """ train the model pipeline """
        super().train(self._start_epoch, self._max_epochs)
