import time

import torch
from loguru import logger
from omegaconf import OmegaConf


class TrainerBase:
    """ Base class for trainers """

    def __init__(self, cfg):

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)

        # epoch
        self._epoch = 0

        # start and max epochs
        self._start_epoch = 0
        self._max_epochs = cfg.scheduler.epochs

        # global training step
        self._global_step = 0

        # scores
        self._last_score = None
        self._best_score = None

        # model
        self._model = None
        self._ema_model = None  # ema model

    def train(self, start_epoch, max_epochs):
        """ train the model pipeline """

        self._epoch = start_epoch
        self._start_epoch = start_epoch
        self._max_epochs = max_epochs

        try:
            # before train
            self.before_train()

            # train loop
            for self._epoch in range(start_epoch, max_epochs):

                # before epoch
                self.before_epoch()

                # train epoch
                self.train_epoch()

                # after epoch
                self.after_epoch()

                # val step
                if self.val_dl:
                    self.val_epoch()

                # test step
                self.test_epoch()

                self._epoch += 1
        #
        except Exception:
            logger.exception("Exception during train the model")
            raise
        #
        finally:
            # after train
            self.after_train()

    def before_train(self):
        """ called before the training loop """
        pass

    def before_epoch(self):
        """ called before each epoch """
        pass

    def train_epoch(self):
        """ called for each epoch """
        # timer
        data_time = time.time()

        #
        for step, (tuples, target) in enumerate(self.train_dl):

            # global step
            self._global_step += 1

            # data_time
            data_time = torch.as_tensor(time.time() - data_time)

            #
            num_imgs = len(tuples[0])

            # batch time
            batch_time = time.time()

            # run
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
                        "batch_time":   batch_time
                    }

                # write
                self.write_metrics(metrics, step, len(
                    self.train_dl), global_step=self._global_step)

                #
                self.writer.add_images(tuple_i, step)
                # self.writer.add_graph(self._model, tuple_i)

            #
            data_time = time.time()

    def after_epoch(self):
        """ called after each epoch """
        pass

    def after_train(self):
        """ called after the training loop """
        pass

    def val_epoch(self):
        """ called for each epoch """

        # set to eval
        if self._model.training:
            self._model.eval()

        # eval
        self.writer.eval()

        # hard mine
        self.refresh_val_data()

        # timer
        data_time = time.time()

        #
        for step, (tuples, target) in enumerate(self.val_dl):

            with torch.no_grad():

                # data_time
                data_time = torch.as_tensor(time.time() - data_time)

                #
                num_imgs = len(tuples[0])

                # batch time
                batch_time = time.time()

                # run
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
                            "batch_time":   batch_time
                        }

                    # write
                    self.write_metrics(metrics, step, len(self.val_dl))

                #
                data_time = time.time()

    def test_epoch(self):
        pass

    # def load_state_dict(self, state_dict):
    #     logger = logging.getLogger(__name__)

    #     self._epoch = state_dict["iteration"]

    #     for key, value in state_dict.get("hooks", {}).items():

    #         for h in self._hooks:
    #             try:
    #                 name = type(h).__qualname__
    #             except AttributeError:
    #                 continue
    #             if name == key:
    #                 h.load_state_dict(value)
    #                 break
    #         else:
    #             logger.warning(
    #                 f"cannot find the hook '{key}', its state_dict is ignored.")
