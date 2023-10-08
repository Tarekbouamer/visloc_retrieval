
from loguru import logger
from omegaconf import OmegaConf


class TrainerBase:
    """ Base class for trainers """

    def __init__(self, cfg):

        # cfg
        self.cfg = cfg if isinstance(cfg, OmegaConf) else OmegaConf.create(cfg)

        # workspace
        self._workspace = None

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
        """ train the model pipeline 
                1. before train
                2. train loop
                    2.1 train epoch
                    2.2 val epoch
                    2.3 test epoch
                3. after train
        """

        self._epoch = start_epoch
        self._start_epoch = start_epoch
        self._max_epochs = max_epochs

        try:
            # before train
            self.before_train()

            # train loop
            for self._epoch in range(start_epoch, max_epochs):

                # train epoch
                self.train_epoch()

                # val step
                if self.val_dl:
                    self.val_epoch()

                # test step
                self.test_epoch()

                self._epoch += 1
        #
        except Exception:
            logger.exception("Exception during train the model")
            raise Exception
        finally:
            # after train
            self.after_train()

    def before_train(self):
        """ called before the training loop """
        raise NotImplementedError

    def train_epoch(self):
        """ called for each epoch """
        raise NotImplementedError

    def after_train(self):
        """ called after the training loop """
        raise NotImplementedError

    def val_epoch(self):
        """ called for each epoch """
        raise NotImplementedError

    def test_epoch(self):
        """ called for each epoch """
        raise NotImplementedError

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
