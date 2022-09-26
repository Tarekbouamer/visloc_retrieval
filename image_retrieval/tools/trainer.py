from ast import arg
import numpy as np
import time
import shutil
import torch

import tensorboardX as tensorboard
import torchvision as torchvision
import os

from timm import utils

# 
from .model             import build_model, run_pca
from .optimizer         import build_optimizer, build_lr_scheduler
from .dataloader        import build_train_dataloader, build_val_dataloader, build_sample_dataloader
from .loss              import build_loss

from .events            import EventWriter
from .evaluation        import DatasetEvaluator


from image_retrieval.utils.logging                 import  _log_api_usage, setup_logger
from image_retrieval.utils.configurations          import make_config, config_to_string
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

    def __init__(self, epoch=0, start_epoch=1, max_epochs=0):
        
        self.epoch          = epoch
        self.start_epoch    = start_epoch
        self.max_epochs     = max_epochs

    def train(self, start_epoch, max_epochs):
        """
          Args:
              start_epoch, max_epochs (int): See docs above
        """

        self.epoch = self.start_epoch = start_epoch
        self.max_epochs = max_epochs

        try:
            self.before_train() 
            
            for self.epoch in range(start_epoch, max_epochs):
                    
                self.before_epoch()
                self.train_epoch()
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
    
    def train_epoch(self):
        """
            Implement the standard training logic described above.
        """
        
        # set to training 
        if not self.model.training:
            self.model.train()
            self.model.apply(set_batchnorm_eval)
        
        self.writer.train()
        # 
        self.refresh_train_data()
        
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
                
                # torchvision.utils.make_grid()
                
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
            
            # ema update
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
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
                self.write_metrics(metrics, step, len(self.train_dl), global_step=self.global_step)
                
                # 
                self.writer.add_images(tuple_i, step)
                # self.writer.add_graph(self.model, tuple_i)

            #
            data_time = time.time()

    def val_epoch(self):
        """
            Implement the standard training logic described above.
        """
        
        # set to eval 
        if self.model.training:
            self.model.eval()
        
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
                    self.write_metrics(metrics, step, len(self.val_dl) )
                
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
        self.writer = self.build_writers(args, cfg)
        
        # build        
        self.model              = self.build_model(cfg)
        self.optimizer          = self.build_optimizer(cfg, self.model)
        self.train_dl           = self.build_train_loader(args, cfg)       
        self.val_dl             = self.build_val_loader(args, cfg)       
        self.loss               = self.build_loss(cfg)     

        # params 
        self.start_epoch  = 1
        self.global_step  = 0
        self.max_epochs   = cfg["scheduler"].getint("epochs")
        self.last_score = None 
        self.best_score = None 
        
        # resume
        if args.resume:
            self.resume_or_load()
        
        # init pca
        elif cfg["global"].getboolean("update"):
            self.init_model()
            
        # ema
        self.model_ema = self.build_ema_model(args, cfg)
        
        # scheduler
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # evaluation
        self.evaluator  = DatasetEvaluator(args, cfg, self.model, self.model_ema, self.get_dataset(), self.writer)
            
        # evaluate starting epoch
        if args.eval:
            self.test()
            
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
        
        # model
        snapshot_last_path = os.path.join(self.args.directory, "last_model.pth.tar")
        logger.info(f"resume load model {snapshot_last_path}")

        # load
        snapshot_last = resume_from_snapshot(self.model, snapshot_last_path, ["body", "head"])
        
        # optimizer
        self.optimizer.load_state_dict(snapshot_last["state_dict"]["optimizer"])
        # print(snapshot_last["state_dict"]["optimizer"])
        # input()
        # scores
        self.start_epoch    = snapshot_last["training_meta"]["epoch"] + 1
        self.best_score     = snapshot_last["training_meta"]["best_score"]
        self.global_step    = snapshot_last["training_meta"]["global_step"]

        # set metrics 
        self.writer.set(snapshot_last["state_dict"])

        del snapshot_last
            
    def before_epoch(self):
        logger.info(f"learning rates {self.scheduler.get_lr()}")
        for it, lr_i in enumerate(self.scheduler.get_lr()):
            self.writer.add_scalar(f'lr/{it}', lr_i, self.epoch)
    
    def after_epoch(self):
        
        snapshot_last = os.path.join(self.args.directory, "last_model.pth.tar".format(self.epoch))
        
        logger.info(f"save snapshot:    {snapshot_last}")
        
        # get metrics
        metrics = self.writer.get()
        meters_out = {k : v.state_dict() for k, v in metrics.items()}

        # save model last
        save_snapshot(snapshot_last, self.cfg, self.epoch, self.last_score, self.best_score, self.global_step,
                        body=self.model.body.state_dict(),
                        head=self.model.head.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        **meters_out
                        )
        
        # save model ema 
        if self.model_ema is not None:
            snapshot_ema = os.path.join(self.args.directory, "ema_model.pth.tar".format(self.epoch))
            
            logger.info(f"save ema snapshot:    {snapshot_ema}")
            
            save_snapshot(snapshot_ema, self.cfg, self.epoch, self.last_score, self.best_score, self.global_step,
                            body=self.model_ema.module.body.state_dict(),
                            head=self.model_ema.module.head.state_dict(),
                            optimizer=self.optimizer.state_dict(),
                            **meters_out
                            )
        
        # update learning rate 
        self.scheduler.step()

    def val_epoch(self):
        if (self.epoch % self.cfg["general"].getint("val_interval")) != 0:
            return

        logger.info(f"val epoch {self.epoch}")
        super().val_epoch()
        
    def test_epoch(self):
        
        if (self.epoch % self.cfg["general"].getint("test_interval")) != 0:
            return
        
        logger.info(f"test epoch {self.epoch}")

        # test
        if self.evaluator is not None:
            self.last_score = self.test()
            
        # load snapshot
        if self.model_ema is not None:
            snapshot_path   = os.path.join(self.args.directory, "ema_model.pth.tar")
            snapshot        = torch.load(snapshot_path, map_location="cpu")
        else:
            snapshot_path   = os.path.join(self.args.directory, "last_model.pth.tar")
            snapshot        = torch.load(snapshot_path, map_location="cpu")

        # update score
        snapshot["training_meta"]["last_score"] = self.last_score
        torch.save(snapshot, snapshot_path)
 
        # save best
        if (self.best_score is None) or \
            (self.last_score >= self.best_score):
                
            best_snapshot_path = os.path.join(self.args.directory, "best_model.pth.tar")
            
            self.best_score = self.last_score
            shutil.copy(snapshot_path, best_snapshot_path)
            logger.info(f"save best snapshot:   {best_snapshot_path}")
    
    def init_model(self):
            
        logger.info("run init pca")
            
        sample_dl = build_sample_dataloader(self.cfg, self.get_dataset())
            
        layer = run_pca(self.args, self.cfg,  self.model, sample_dl)
            
        # save layer to whithen_path
        layer_path = os.path.join(self.args.directory, "whiten.pth")
        logger.info(f"save whiten layer: {layer_path}")
        torch.save(layer.state_dict(), layer_path)
            
        # init
        self.model.head.whiten.load_state_dict(layer.state_dict())
            
        logger.info("pca done")
        
    def before_train(self):
        
        pass
                    
    def train(self):
        """
          Run training.
          Returns:
              OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        logger.info("start training")
        super().train(self.start_epoch, self.max_epochs)
        
    def train_epoch(self):
        logger.info(f"train epoch {self.epoch}")
        super().train_epoch()

    def state_dict(self):
        ret             = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])
    
    def refresh_train_data(self):
        logger.info(f"refresh train dataset")
        stats = self.train_dl.dataset.create_epoch_tuples(self.cfg, self.model) 
        
        if len(stats) > 0:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    self.writer.add_scalar(k, v, self.epoch)
                else:
                    self.writer.add_scalar(k, torch.as_tensor(v), self.epoch)
    
    def refresh_val_data(self):
        logger.info(f"refresh val dataset")
        stats = self.val_dl.dataset.create_epoch_tuples(self.cfg, self.model)
        
        if len(stats) > 0:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    self.writer.add_scalar(k, v, self.epoch)
                else:
                    self.writer.add_scalar(k, torch.as_tensor(v), self.epoch)
                
    def write_metrics(self, metrics, step, max_steps, global_step=None, prefix=""):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
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
            self.writer.log(self.epoch, self.max_epochs,  step, max_steps)

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
    
    def build_ema_model(self, args, cfg):

        ema_model = utils.ModelEmaV2(self.model, decay=cfg["body"].getfloat("ema_decay"))
        
        logger.info(f"ema_model:\n {ema_model}")
        
        if args.resume:
            # model
            snapshot_ema_path = os.path.join(self.args.directory, "ema_model.pth.tar")
            logger.info(f"resume load ema model {snapshot_ema_path}")

            # load
            snapshot_last = resume_from_snapshot(ema_model.module, snapshot_ema_path, ["body", "head"])
        
            self.best_score = snapshot_last["training_meta"]["best_score"]

            del snapshot_last

        return ema_model

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
    def build_val_loader(self, args, cfg):
        """
        Returns:
        """
        return build_val_dataloader(args, cfg)    
    
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
    def build_writers(self, args, cfg):        
        return EventWriter(args.directory, 
                           cfg["general"].getint("log_interval"))
        
    def test(self):
                
        logger.info(f"evaluate epoch {self.epoch}")
              
        # test
        results_dict = self.evaluator.evaluate(self.epoch)
            
        # overall score
        scores = [v for k, v in  results_dict.items()]
        score = sum(scores)/len(scores)
        
        logger.info(f"score: {score}")
            
        #   
        return score