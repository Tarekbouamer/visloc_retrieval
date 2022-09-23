from collections import OrderedDict

import torch
from torch import nn

import os
from .test import  build_paris_oxford_dataset, test_asmk, test_global_descriptor
import image_retrieval.utils.evaluation.asmk as eval_asmk


# logger
import logging
logger = logging.getLogger("retrieval")

class DatasetEvaluator:
    """
      Base class for a dataset evaluator.
    """
    def build_dataset(self):
        pass

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:
        .. code-block:: python
            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluator(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, args, cfg, model, train_dataset, writer):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        
        #  model
        self.model = model
        self.train_dataset = train_dataset
        
        self.writer = writer
        
        #  
        self.test_mode      = cfg["test"].get("mode")
        self.test_datasets  = cfg["test"].getstruct("datasets")
        self.multi_scale    = cfg["test"].getboolean("multi_scale")
        
        self.descriptor_size = cfg["global"].getint("global_dim")

        # 
        self.cfg    = cfg
        self.args   = args
        
        # asmk
        if self.test_mode == "asmk":
            
            # number of sampled image 
            self.num_samples = cfg["test"].getint("num_samples")
            
            # train and save the codebook for each test set
            self.build_codebook()
        
        # 
        logger.info(f"init evaluator on ({self.test_mode}) mode")
     
    def build_codebook(self, ):
        
        # eval mode
        if self.model.training:
            self.model.eval()

        logger.info('init asmk')
        asmk, params = eval_asmk.asmk_init()
        
        # train codebook
        save_path =  os.path.join(self.args.directory, self.cfg["dataloader"].get("dataset") + "_codebook.pkl")
        logger.info(f'train codebook:   {save_path}')
 
        # sample data    
        idxs = torch.randperm(len(self.train_dataset))[ :self.num_samples]
        train_images  = [self.train_dataset[i] for i in idxs] 

        # Run train_codebook
        self.asmk = eval_asmk.train_codebook(self.cfg, train_images, self.model, asmk,
                                        save_path=save_path)
        
        return asmk
        
    def build_test_dataset(self, data_path, dataset):
        
        query_dl, database_dl, ground_truth = None, None, None
                    
        if dataset in ['roxford5k', 'rparis6k', "val_eccv20"]:
            query_dl, database_dl, ground_truth = build_paris_oxford_dataset(data_path, dataset, self.cfg)
            
        return query_dl, database_dl, ground_truth
    
    def write_metrics(self, metrics, datatset, step, scale=1):

        # push metrics to history
        if len(metrics) > 1:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    self.writer.put(datatset +"/"+ k, v * scale)
                else:
                    self.writer.put(datatset +"/"+ k, torch.as_tensor(v * scale))
                    
        # write to board
        self.writer.write(step)
        

    def evaluate(self, epoch):
        
        # eval mode
        if self.model.training:
            self.model.eval()
        #
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
            if self.test_mode == "global":
                metrics = test_global_descriptor(dataset, query_dl, database_dl, self.model, self.descriptor_size, ground_truth)
            
            elif self.test_mode == "asmk":
                metrics = test_asmk(dataset, query_dl, database_dl, self.model, self.descriptor_size, ground_truth, self.asmk)
            
            else:
                logger.error(f"{self.test_mode} is not implemented yet")
                
            # write
            self.write_metrics(metrics, dataset, epoch, scale=100)
            
            # map
            results[dataset] = metrics["map"]

       
        return results