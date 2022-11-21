from collections import OrderedDict

import torch
from torch import nn

import os
from retrieval.test import  build_paris_oxford_dataset, test_asmk, test_global_descriptor
import retrieval.utils.evaluation.asmk as eval_asmk

from retrieval.feature_extractor import FeatureExtractor

# logger
import logging

logger = logging.getLogger("retrieval")

class DatasetEvaluator:
    """
      Base class for a dataset evaluator.
    """
    def __init__(self, args, cfg, model, model_ema=None, writer=None):
        
        # mode
        self.test_mode = cfg['test'].get('mode')
        
        # features extractor
        model = model if model_ema is None else model_ema.module
        self.feature_extractor = FeatureExtractor(model=model, cfg=cfg)
                
        # writer
        self.writer = writer
        
        #  
        self.test_datasets  = cfg["test"].getstruct("datasets")
        self.scales    = cfg["test"].getboolean("multi_scale")
        
        #
        self.descriptor_size = cfg["global"].getint("global_dim")

        # 
        self.cfg    = cfg
        self.args   = args  

    def write_metrics(self, metrics, datatset, step, scale=1):
        if self.writer is None:
            return
        
        # push metrics to history
        if len(metrics) > 0 :
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    self.writer.put(datatset +"/"+ k, v * scale)
                else:
                    self.writer.put(datatset +"/"+ k, torch.as_tensor(v * scale))
                    
        # write to board
        self.writer.write(step)
        
    def evaluate(self, epoch=0):
        """
        """
        raise NotImplementedError


class GlobalEvaluator(DatasetEvaluator):
    """

    """

    def __init__(self, args, cfg, model, model_ema=None, writer=None, **kwargs):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__(args, cfg, model, model_ema, writer)
        
        logger.info(f"init evaluator on ({self.test_mode}) mode")
          
    def build_test_dataset(self, data_path, dataset):
        
        query_dl, database_dl, ground_truth = None, None, None
                    
        if dataset in ['roxford5k', 'rparis6k', "val_eccv20"]:
            query_dl, database_dl, ground_truth = build_paris_oxford_dataset(data_path, dataset, self.cfg)
        else:
            raise KeyError
            
        return query_dl, database_dl, ground_truth
            
    def evaluate(self, epoch=0, scales=[1]):
        
        # eval mode
        self.feature_extractor.eval()
        
        #
        if self.writer is not None:
            self.writer.test()
        
        # data path
        if not os.path.exists(self.args.data):
            logger.error("path not found: {self.args.data}")   
        
        data_path = self.args.data 
        
        # result dictionary
        results = OrderedDict()

        # eval all test_datasets
        for dataset in self.test_datasets:
            
            # build dataset
            query_dl, database_dl, ground_truth = self.build_test_dataset(data_path, dataset)
            
            # test
            metrics = test_global_descriptor(dataset, query_dl, database_dl, 
                                             self.feature_extractor, self.descriptor_size, 
                                             ground_truth,
                                             scales=scales)
                
            # write
            self.write_metrics(metrics, dataset, epoch, scale=100)
            
            # map
            results[dataset] = metrics["map"]
            
        return results
    
    
class ASMKEvaluator(DatasetEvaluator):
    
    def __init__(self, args, cfg, model, model_ema=None, writer=None, **kwargs):
        super().__init__(args, cfg, model, model_ema, writer)

        # train dataset
        self.train_dataset = kwargs.pop('train_dataset', None)
         
        # number of sampled image 
        self.num_samples = cfg["test"].getint("num_samples")
                
        # train and save the codebook for each test set
        self.build_codebook()
        
        #
        logger.info(f"init evaluator on ({self.test_mode}) mode")
        
    def build_codebook(self, ):
        
        # eval mode
        self.feature_extractor.eval()

        logger.info('init asmk')
        asmk, params = eval_asmk.asmk_init()
        
        # train codebook
        save_path =  os.path.join(self.args.directory, self.cfg["dataloader"].get("dataset") + "_codebook.pkl")
 
        idxs = torch.randperm(len(self.train_dataset))[ :self.num_samples]
        train_images  = [self.train_dataset[i] for i in idxs] 
        
        logger.info(f'train codebook {len(train_images)} :   {save_path}')

        # train_codebook
        self.asmk = eval_asmk.train_codebook(self.cfg, train_images, self.feature_extractor, asmk)
        
        return asmk
    
    def build_test_dataset(self, data_path, dataset):
        
        query_dl, database_dl, ground_truth = None, None, None
                    
        if dataset in ['roxford5k', 'rparis6k', "val_eccv20"]:
            query_dl, database_dl, ground_truth = build_paris_oxford_dataset(data_path, dataset, self.cfg)
        else:
            raise KeyError
            
        return query_dl, database_dl, ground_truth    
    
    def evaluate(self,epoch=0, scales=[1]):
        
        # eval mode
        self.feature_extractor.eval()
        
        #
        if self.writer is not None:
            self.writer.test()
        
        # data path
        if not os.path.exists(self.args.data):
            logger.error("path not found: {self.args.data}")   
        
        data_path = self.args.data 
        
        # result dictionary
        results = OrderedDict()

        # eval all test_datasets
        for dataset in self.test_datasets:
            
            # build dataset
            query_dl, database_dl, ground_truth = self.build_test_dataset(data_path, dataset)
            
            # test
            metrics = test_asmk(dataset, query_dl, database_dl, 
                                self.feature_extractor, 
                                self.descriptor_size, ground_truth, self.asmk)
                
            # write
            self.write_metrics(metrics, dataset, epoch, scale=100)
            
            # map
            results[dataset] = metrics["map"]

       
        return results
    

def build_evaluator(args, cfg, model, model_ema, writer, **meta):
    
    if cfg['test'].get('mode') == 'global':
        return GlobalEvaluator(args, cfg, model, model_ema, writer, **meta)
    
    elif cfg['test'].get('mode') == 'asmk':
        return ASMKEvaluator(args, cfg, model, model_ema, writer, **meta)
    
    else:
        raise KeyError

    

