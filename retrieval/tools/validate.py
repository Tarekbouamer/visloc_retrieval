
import time

from collections import OrderedDict

import torch

# core
from core.utils.general     import humanbytes
from core.utils.meters      import AverageMeter
from core.utils             import logging



def validate(model, config, dataloader, logger, **varargs):

    #
    criterion   = varargs["criterion"]

    # create tuples for validation
    data_config = config["dataloader"]

    avg_neg_distance = dataloader.dataset.create_epoch_tuples(model, 
                                                              logger=logger,
                                                              out_dim=varargs["out_dim"],
                                                              world_size=varargs["world_size"],
                                                              rank=varargs["rank"],
                                                              device=varargs["device"],
                                                              config=config)
    
    # Free Gpu cache
    # distributed.barrier()
    torch.cuda.empty_cache()
    
    logger.debug("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
    logger.debug("Cached:    " + humanbytes(torch.cuda.memory_reserved()))

    # Switch to eval mode
    model.eval()
    # dataloader.batch_sampler.set_epoch(varargs["epoch"])


    loss_meter = AverageMeter(())
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    data_time = time.time()

    for it, batch in enumerate(dataloader):
  
        with torch.no_grad():

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()
            
            # Over tuples
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

                    pred =  model(img=item.cuda(), do_loss=True, do_whitening=True)
                
                    vecs[item_idx:item_idx+1, :] = pred["descs"]
                
                # compute loss
                loss = criterion(vecs, target.cuda())

                # Accumulate
                losses["loss"] += (1/num_tuples) * loss.mean()

            # Update meters
            loss_meter.update(losses["loss"].cpu())
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            del tuple, target, neg_nums
            del loss, losses

        # Log batch
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                logger,
                varargs["summary"], "val", varargs["global_step"],
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(dataloader),
                OrderedDict([
                    ("loss", loss_meter),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)
                ])
            )

        data_time = time.time()

    return loss_meter.mean
