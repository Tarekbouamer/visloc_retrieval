# General
import argparse
from os import makedirs, path
import shutil
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

import tensorboardX as tensorboard

# Core
from core.utils.misc        import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, NORM_LAYERS, OTHER_LAYERS

from core.utils             import logging
from image_retrieval.utils.snapshot    import save_snapshot, resume_from_snapshot, pre_train_from_snapshots
from core.utils.meters      import AverageMeter

# Image Retrieval
from image_retrieval.tools import make_dataloader, make_model_resnext, make_optimizer
from image_retrieval.tools import train, validate, test

from image_retrieval.configuration              import load_config, config_to_string, DEFAULTS as DEFAULT_CONFIGS

def log_debug(msg, *args, **kwargs):
    # if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    # if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved',
                        default='./experiments/')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved',
                        default='/media/dl/Data/datasets')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./image_retrieval/configuration/defaults/resneck.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run",
                        default=True)

    parser.add_argument('--resume', metavar='FILENAME', type=str,
                        help='name of the latest checkpoint (default: None)',
                        default='./experiments/retrieval-SfM-120k_resneck34_triplet_m0.50_GeM_Adam_lr5.0e-07_wd1.0e-06_nnum5_bsize5_uevery1_imsize1024/model_last.pth.tar')



    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")


    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def make_dir(config, directory):
    # Create export dir name if it doesnt exist in your experiment folder
    extension = "{}".format(config["dataloader"].get("training_dataset"))
    extension += "_{}".format(config["body"].get("arch"))

    extension += "_{}_m{:.2f}".format(config["global"].get("loss"),
                                      config["global"].getfloat("loss_margin"))

    if config["global"].getstruct("pooling"):
        extension += "_{}".format(config["global"].getstruct("pooling")["name"])
        
    
    if config["global"].get("attention"):
        extension += "_{}_encs{}_h{}".format(config["global"].get("attention"),
                                            config["global"].getfloat("num_encs"),
                                            config["global"].getfloat("num_heads"))

    extension += "_{}_lr{:.1e}_wd{:.1e}".format(config["optimizer"].get("type"),
                                                config["optimizer"].getfloat("lr"),
                                                config["optimizer"].getfloat("weight_decay"))

    extension += "_nnum{}".format(config["dataloader"].getint("neg_num"))
    
    extension += "_bsize{}_uevery{}_imsize{}".format(config["dataloader"].getint("train_batch_size"),
                                                     config["dataloader"].getint("update_every"),
                                                     config["dataloader"].getint("train_longest_max_size"))

    directory = path.join(directory, extension)


    if not path.exists(directory):
        log_debug("Create experiment path  from %s", directory)
        makedirs(directory)

    return directory


def make_config(args):

    
    
    log_info("Loading configuration from %s", args.config)

    config = load_config(args.config, DEFAULT_CONFIGS["default"])

    log_info("\n%s", config_to_string(config))

    return config


def main(args):

    # Initialize multi-processing
    print(args.local_rank)
    print(args.config)

    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = 0, 1

    # Set device
    torch.cuda.set_device(device_id)

    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # Load configuration
    config = make_config(args)

    # Experiment Path TODO:
    exp_dir = make_dir(config, args.directory)

    # Initialize logging
    if rank == 0:
        logging.init(exp_dir, "training" if not args.eval else "eval")
        summary = tensorboard.SummaryWriter(args.directory)
    else:
        summary = None
    
    log_info("\n%s", config_to_string(config))

    body_config = config["body"]
    optimizer_config = config["optimizer"]

    # Load data
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size,
                                                       log_debug=log_debug,
                                                       log_info=log_info)

    # Initialize model
    if body_config.getboolean("pretrained"):
        log_debug("Use pre-trained model %s", body_config.get("arch"))
    else:
        log_debug("Initialize model to train from scratch %s". body_config.get("arch"))

    # Load model
    model, output_dim = make_model_resnext(args, config, train_dataloader, 
                                        rank=rank, 
                                        world_size=world_size, 
                                        device=device,
                                        log_debug=log_debug,
                                        log_info=log_info)

    print(model)
    
    # Resume / Pre_Train
    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        log_debug("Loading snapshot from %s", args.resume)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "ret_head"])
    
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_debug("Loading pre-trained model from %s", args.pre_train)
        pre_train_from_snapshots(model, args.pre_train, ["body", "ret_head"])
    
    else:
        # assert not args.eval, "--resume is needed in eval mode"
        snapshot = None

    
    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = model.cuda(device)
        
    # Create optimizer & scheduler
    optimizer, scheduler, parameters, batch_update, total_epochs = make_optimizer(model, config, 
                                                                                  epoch_length=len(train_dataloader))
   
    if args.resume:
        optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])

    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    meters = {
        "loss": AverageMeter((), momentum),
        "ret_loss": AverageMeter((), momentum)
    }

    if args.resume:
        start_epoch = snapshot["training_meta"]["epoch"] + 1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]

        for name, meter in meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        start_epoch = 0
        best_score = {
            "val": 1000.0,
            "test": 0.0,
        }
        global_step = 0

    # Optional: evaluation only:
    if args.eval:
        log_info("Evaluation epoch %d", start_epoch - 1)

        test(args, config, model, 
             train_imgs=train_dataloader.dataset.images,
             rank=rank, 
             world_size=world_size,
             output_dim=output_dim,
             device=device,
             log_debug=log_debug,
             log_info=log_info)


        log_info("Evaluation Done ..... ")

        # Exit
        exit(0)

    for epoch in range(start_epoch, total_epochs):

        log_info("Starting epoch %d", epoch + 1)

        if not batch_update:
            scheduler.step(epoch)

        score = {}
        
        # Run training
        global_step = train(model, config, train_dataloader, optimizer, scheduler, meters,
                            summary=summary,
                            batch_update=batch_update,
                            log_interval=config["general"].getint("log_interval"),
                            epoch=epoch,
                            num_epochs=total_epochs,
                            global_step=global_step,
                            output_dim=output_dim,
                            world_size=world_size,
                            rank=rank,
                            device=device,
                            loss_weights=optimizer_config.getstruct("loss_weights"),
                            log_debug=log_debug,
                            log_info=log_info
                            )

        # Save val snapshot (only on rank 0)
        if rank == 0 and (epoch + 1) % config["general"].getint("val_interval") == 0:
            snapshot_file = path.join(exp_dir, "model_{}.pth.tar".format(epoch))

            log_debug("Saving snapshot to %s", snapshot_file)

            meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}

            save_snapshot(  snapshot_file, config, epoch, 0, best_score, global_step,
                            body=model.body.state_dict(),
                            ret_head=model.ret_head.state_dict(),
                            optimizer=optimizer.state_dict(),
                            **meters_out_dict)

        # Run validation
        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            log_info("Validating epoch %d", epoch + 1)

            # Validate
            score['val'] = validate(model, config, val_dataloader,
                                    summary=summary,
                                    batch_update=batch_update,
                                    log_interval=config["general"].getint("log_interval"),
                                    epoch=epoch,
                                    num_epochs=total_epochs,
                                    global_step=global_step,
                                    output_dim=output_dim,
                                    world_size=world_size,
                                    rank=rank,
                                    device=device,
                                    loss_weights=optimizer_config.getstruct("loss_weights"),
                                    log_debug=log_debug,
                                    log_info=log_info
                                    )

        # Run Test
        if (epoch + 1) % config["general"].getint("test_interval") == 0:
            log_info("Testing epoch %d", epoch + 1)
            
            # Test
            score['test'] = test(args, config, model, rank=rank, world_size=world_size,
                                 output_dim=output_dim,
                                 device=device,
                                 log_debug=log_debug,
                                 log_info=log_info)

            # Update the score on the last saved snapshot
            if rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot
            
            # Save Best
            if score['test'] > best_score['test']:
                best_score = score
                if rank == 0:
                    shutil.copy(snapshot_file, path.join(exp_dir, "test_model_best.pth.tar"))

        # Save last snapshot
        if rank == 0 :
            snapshot_file = path.join(exp_dir, "model_last.pth.tar")

            log_debug("Saving snapshot to %s", snapshot_file)

            meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}

            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                            body=model.body.state_dict(),
                            ret_head=model.ret_head.state_dict(),
                            optimizer=optimizer.state_dict(),
                            **meters_out_dict)


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())