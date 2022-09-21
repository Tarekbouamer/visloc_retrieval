from calendar import c
import time
from collections import OrderedDict

import torch
import torch.nn as nn 

# core 
from core.utils.meters      import AverageMeter
from core.utils.general     import humanbytes

from core.utils             import logging

        
def set_batchnorm_eval(m):
    classname = m.__class__.__name__

    if classname.find('BatchNorm') != -1:
        m.eval()
        
    if classname.find('ABN') != -1:
        m.eval()


def train(model, config, dataloader, optimizer, scheduler, meters, logger, **varargs):
    
    criterion   = varargs["criterion"]
    optim_cfg   = config["optimizer"]
    
    # Create tuples for training
    avg_neg_distance = dataloader.dataset.create_epoch_tuples(model, 
                                                              logger=logger,
                                                              out_dim=varargs["out_dim"],
                                                              world_size=varargs["world_size"],
                                                              rank=varargs["rank"],
                                                              device=varargs["device"],
                                                              config=config)

    # Free Gpu cache
    torch.cuda.empty_cache()
    
    logger.debug("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
    logger.debug("Cached:    " + humanbytes(torch.cuda.memory_reserved()))

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)
    
    # dataloader.batch_sampler.set_epoch(varargs["epoch"])
    optimizer.zero_grad()
    
    global_step = varargs["global_step"]

    data_time_meter     = AverageMeter((), meters["loss"].momentum)
    batch_time_meter    = AverageMeter((), meters["loss"].momentum)

    data_time = time.time()

    for it, batch in enumerate(dataloader):
    

        # Measure data loading time
        data_time_meter.update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        losses = {  "loss": 0.0 }
        num_tuples = len(batch["q"])
          
        # Run network for each tuple
        for _, (q, p, ns, target, neg_nums) in enumerate(zip(batch["q"], batch["p"], batch["ns"], batch["target"], batch["neg_nums"])):
            
            # prepare data as list 
            tuple =  model._prepare_tuple(q, p, ns)
            
            vecs = torch.zeros(len(tuple), varargs["out_dim"]).cuda()
            
            # extract vectors
            for item_idx, item in enumerate(tuple):
                
                item = item.unsqueeze(0)
                pred =  model(img=item.cuda(), do_whitening=True)
                vecs[item_idx:item_idx+1, :] = pred["descs"]
            
            # compute loss
            loss = criterion(vecs, target.cuda())
            loss.backward()
                        
            # Accumulate
            losses["loss"] += (1/num_tuples) * loss.mean()

        # grad clipping
        if optim_cfg.get("grad_mode")   == 'norm':
            nn.utils.clip_grad_norm_( model.parameters() , 
                                         max_norm=optim_cfg.get("grad_value"), 
                                         norm_type=optim_cfg.get("grad_type"))
            
        # optimize for the actual effective batch :) 
        optimizer.step()
        optimizer.zero_grad()

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())

        batch_time_meter.update(torch.tensor(time.time() - batch_time))

        # Clean-up
        del tuple, target, neg_nums

        # Log
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                logger,
                varargs["summary"], "train", global_step,
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(dataloader),
                OrderedDict([
                    ("lr_body", scheduler.get_lr()[1] * 1e6),
                    ("lr_ret",  scheduler.get_lr()[2] * 1e6),
                    ("loss", meters["loss"]),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)
                ])
            )

        data_time = time.time()

    return global_step
