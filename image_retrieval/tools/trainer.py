import numpy as np
import time
import shutil
import torch

import tensorboardX as tensorboard

import os

# 
from .model             import build_model, run_pca
from .optimizer         import build_optimizer, build_lr_scheduler
from .dataloader        import build_train_dataloader, build_sample_dataloader
from .loss              import build_loss

from .events            import EventWriter
from .evaluation        import DatasetEvaluator


from core.utils.logging                 import  _log_api_usage, setup_logger
from core.utils.configurations          import make_config, config_to_string
from image_retrieval.utils.snapshot     import save_snapshot, resume_from_snapshot, pre_train_from_snapshots


# logger
import logging
logger = logging.getLogger("retrieval")

# modes
TEST_MODES = ["global", "asmk", "all"]


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
            
class TrainerBase:
    """
      Base class for iterative trainer with hooks.
      The only assumption we made here is: the training runs in a loop.
      A subclass can implement what the loop is.
      We made no assumptions about the existence of dataloader, optimizer, model, etc.
      Attributes:
          iter(int): the current iteration.
          start_epoch(int): The iteration to start with.
              By convention the minimum possible value is 0.
          max_epochs(int): The iteration to end training.
          storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self, epoch=-1, start_epoch=1, max_epochs=0):
        
        self.epoch          = epoch
        self.start_epoch    = start_epoch
        self.max_epochs     = max_epochs

    def train(self, start_epoch, max_epochs):
        """
          Args:
              start_epoch, max_epochs (int): See docs above
        """

        logger.info(f"starting training {start_epoch}")

        self.epoch = self.start_epoch = start_epoch
        self.max_epochs = max_epochs

        try:
            self.before_train() 
            
            for self.epoch in range(start_epoch, max_epochs):
                    
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
                self.val_epoch()
                self.test_epoch()
                    
                self.epoch += 1
        #
        except Exception:
            logger.exception("exception during training:")
            raise
        #
        finally:
            self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass
    
    def test_epoch(self):
        pass
    
    def run_epoch(self):
        """
            Implement the standard training logic described above.
        """
        
        # set to training 
        if not self.model.training:
            self.model.train()
        
        self.model.apply(set_batchnorm_eval)

        # zero grad
        self.optimizer.zero_grad()
        
        # timer
        data_time = time.time()
        
        # 
        for step, (tuples, target) in enumerate(self.train_dl):
                        
            # global step
            self.global_step += 1
            
            # data_time
            data_time = torch.as_tensor(time.time() - data_time)

            # 
            num_imgs  = len(tuples[0])
            
            # batch time
            batch_time = time.time()
            
            # run
            for tuple_i, target_i in zip(tuples, target):
                
                vecs = torch.zeros(num_imgs, self.cfg["global"].getint("global_dim")).cuda()
                
                # extract vectors
                for n in range(len(tuple_i)):

                    pred = self.model(tuple_i[n].cuda(), do_whitening=True)
                    vecs[n, :]  = pred
    
                # compute loss 
                loss = self.loss(vecs, target_i.cuda())
                
                # backward
                loss.backward()
            
            # optimize 
            self.optimizer.step()
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
                self.write_metrics(metrics, step)
            
            #
            data_time = time.time()

    def state_dict(self):
        ret = {"iteration": self.epoch}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        
        self.epoch = state_dict["iteration"]
        
        for key, value in state_dict.get("hooks", {}).items():
            
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"cannot find the hook '{key}', its state_dict is ignored.")


class ImageRetrievalTrainer(TrainerBase):
    def __init__(self, args, cfg):
        super().__init__()
            
        # cfg 
        logger.info("\n %s", config_to_string(cfg))
        self.cfg          = cfg
        self.args         = args
        
        # writer
        self.writer     = EventWriter(args.directory)
        
        # build        
        self.model              = self.build_model(cfg)
        self.optimizer          = self.build_optimizer(cfg, self.model)
        self.train_dl           = self.build_train_loader(args, cfg)       
        self.val_dl             = self.build_val_loader(args, cfg)       
        self.loss               = self.build_loss(cfg)                
         
        # scheduler
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        #  
        self.start_epoch  = 1
        self.global_step  = 0
        self.max_epochs   = cfg["scheduler"].getint("epochs")
        
        #
        self.compute_pca  = cfg["global"].getboolean("update")
        
        # evaluation
        self.last_score = None 
        self.best_score = None 
        self.evaluator  = DatasetEvaluator(args, cfg, self.model, self.get_dataset())
        
        # resume 
        if args.resume:
            self.resume_or_load()
            self.compute_pca = False 
            
        # evaluate starting epoch
        if args.eval:
            self.before_train()
            self.test()
            self.compute_pca = False
            
        logger.info("init trainer")
        
    def get_dataset(self):
        if self.train_dl is not None:
            return self.train_dl.dataset.images
        
    def resume_or_load(self, resume=True):
        """
            If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
            a `last_checkpoint` file), resume from the file. Resuming means loading all
            available states (eg. optimizer and scheduler) and update iteration counter
            from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
            Otherwise, this is considered as an independent training. The method will load model
            weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
            from iteration 0.
            Args:
                resume (bool): whether to do resume or not
        """
        logger.info(f"resume: {self.args.resume}")
        
        # model
        snapshot = resume_from_snapshot(self.model, 
                                        self.args.resume, 
                                        ["body", "ret_head"])

        # optimizer
        self.optimizer.load_state_dict(snapshot["state_dict"]["optimizer"], strict=True)
        
        # scores
        self.start_epoch    = snapshot["training_meta"]["epoch"] + 1
        self.best_score     = snapshot["training_meta"]["best_score"]
        self.global_step    = snapshot["training_meta"]["global_step"]

        # set metrics 
        self.writer.set(snapshot["state_dict"])

        del snapshot
            
    def before_epoch(self):
        self.refresh_data()
        logger.info(f"learning rates {self.scheduler.get_last_lr()}")
    
    def after_epoch(self):
        
        snapshot_last = os.path.join(self.args.directory, "last_model_.pth.tar".format(self.epoch))
        
        logger.info(f"save snapshot:    {snapshot_last}")
        
        # get metrics
        metrics = self.writer.get()
        meters_out = {k : v.state_dict() for k, v in metrics.items()}

        # save last
        save_snapshot(snapshot_last, self.cfg, self.epoch, self.last_score, self.best_score, self.global_step,
                        body=self.model.body.state_dict(),
                        head=self.model.head.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        **meters_out
                        )
        
        # update learning rate 
        self.scheduler.step()

    def val_epoch(self):
        pass
        
    def test_epoch(self):
        
        if (self.epoch % self.cfg["general"].getint("test_interval")) != 0:
            return
        
        snapshot_last = os.path.join(self.args.directory, "last_model_.pth.tar")

        # test
        if self.evaluator is not None:
            self.last_score = self.test()
            
        #
        snapshot = torch.load(snapshot_last, map_location="cpu")
        snapshot["training_meta"]["last_score"] = self.last_score
        torch.save(snapshot, snapshot_last)
            
        # save best
        if (self.best_score is None) or \
            (self.last_score >= self.best_score):
                
            best_snapshot = os.path.join(self.args.directory, "best_model.pth.tar")
            
            self.best_score = self.last_score
            shutil.copy(snapshot_last, best_snapshot)
            logger.info(f"save best snapshot:   {best_snapshot}")

    def before_train(self):
        
        if self.compute_pca:
            
            logger.info("run PCA")
            
            sample_dl = build_sample_dataloader(self.cfg, self.get_dataset())
            
            layer = run_pca(self.args, self.cfg,  self.model, sample_dl)
            
            # save layer to whithen_path
            layer_path = os.path.join(self.args.directory, "whiten.pth")
            logger.info(f"save whiten layer: {layer_path}")
            torch.save(layer.state_dict(), layer_path)
            
            # init
            self.model.head.whiten.load_state_dict(layer.state_dict())
            
            logger.info("PCA Done")
                    
    def train(self):
        """
          Run training.
          Returns:
              OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        logger.info("start training")
        super().train(self.start_epoch, self.max_epochs)
        
    def run_epoch(self):
        logger.info("run step")
        super().run_epoch()

    def state_dict(self):
        ret             = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])
    
    def refresh_data(self):
        logger.info(f"refresh dataset {self.epoch}")
        self.train_dl.dataset.create_epoch_tuples(self.cfg, self.model) 
    
    def write_metrics(self, metrics, step, prefix=""):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        
        metrics_dict = {k: v.detach().cpu() for k, v in metrics.items()}

        # push metrics to history
        if len(metrics_dict) > 1:
            for k, v in metrics_dict.items():
                if isinstance(v, torch.Tensor):
                    self.writer.put(k, v)
                    
        # write to board
        self.writer.write(self.global_step)
        
        # logger iteration
        if step % self.writer.print_freq == 0:       
            self.writer.log(self.epoch, self.max_epochs,  step, len(self.train_dl))

    @classmethod
    def build_model(cls, cfg):
        """
          Returns:
              torch.nn.Module:
          It now calls :func:`detectron2.modeling.build_model`.
          Overwrite it if you'd like a different model.
        """
        meta_arch  = cfg["body"].get("arch")
        
        model = build_model(cfg)
        
        model.to(torch.device("cuda"))
        _log_api_usage("modeling.meta_arch." + meta_arch)
        
        logger.info(f"model:\n {model}")
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
          Returns:
              torch.optim.Optimizer:
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(self, args, cfg):
        """
        Returns:
        """
        return build_train_dataloader(args, cfg)
    
    @classmethod
    def build_loss(self, cfg):
        """
        Returns:
        """
        return build_loss(cfg)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # TODO: build test data;loader
        pass

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None
        """
        raise NotImplementedError(
            """
            If you want DefaultTrainer to automatically run evaluation,
            please implement `build_evaluator()` in subclasses (see train_net.py for example).
            Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
            )

    def test(self):
                
        logger.info(f"evaluate epoch {self.epoch}")
              
        # test
        results_dict = self.evaluator.evaluate()
            
        # overall score
        scores = [v for k, v in  results_dict.items()]
        score = sum(scores)/len(scores)
            
        #   
        return score