# General
import argparse
from os import makedirs, path
import shutil
import time
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tensorboardX as tensorboard

# Core
import core.backbones   as models
from core.backbones.url     import model_urls, model_urls_cvut
from core.backbones.util    import load_state_dict_from_url, init_weights
from core.utils.download    import download_train, download_test
from core.utils.misc        import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, NORM_LAYERS, OTHER_LAYERS
from core.utils.general     import htime
from core.utils.options     import test_datasets_names
from core.utils             import logging
from image_retrieval.utils.snapshot    import save_snapshot, resume_from_snapshot, pre_train_from_snapshots
from core.utils.meters      import AverageMeter
from core.utils.PCA         import PCA, PCA_whitenlearn_shrinkage
from core.utils.evaluation.ParisOxfordEval import compute_map_and_print

# Image Retrieval
from image_retrieval.configuration                      import load_config, config_to_string, DEFAULTS as DEFAULT_CONFIGS
from image_retrieval.datasets.tuples                    import TuplesDataset, TuplesTransform
from image_retrieval.datasets.generic.generic           import ImagesFromList, ImagesTransform, INPUTS
from image_retrieval.datasets.benchmark                 import ParisOxfordTestDataset

from image_retrieval.datasets.misc                      import iss_collate_fn, collate_tuples

from image_retrieval.modules.heads.attention_head       import AttentionHead

from image_retrieval.algos.attention_algo               import globalFeatureAlgo, globalFeatureLoss
from image_retrieval.models.base                      import ImageRetrievalNet


def set_batchnorm_eval(m):
    classname = m.__class__.__name__

    if classname.find('BatchNorm') != -1:
        m.eval()
        
    if classname.find('ABN') != -1:
        m.eval()


def log_debug(msg, *args, **kwargs):
    # if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    # if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def humanbytes(B):
    """
            Return the given bytes as a human friendly KB, MB, GB, or TB string
            
    """
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)
    
    
def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./cirtorch/configuration/defaults/global_config.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run")

    parser.add_argument('--resume', metavar='FILENAME', type=str,
                        help='name of the latest checkpoint (default: None)')

    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")

    parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                        help='comma separated list of test datasets: ' + ' | '.join(
                            test_datasets_names) + ' (default: roxford5k,rparis6k)')

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
    

    
    extension += "_{}".format(config["global"].getstruct("pooling")["name"])

    extension += "_{}_h{}".format(config["global"].get("attention"),
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


def make_dataloader(args, config, rank=None, world_size=None):
    general_config = config["general"]
    data_config = config["dataloader"]

    # Manually check if there are unknown test datasets
    for dataset in data_config.getstruct("test_datasets"):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

    # Check if test datasets are available, download it if  not !
    name = data_config.get("training_dataset")
    
    if name.startswith('retrieval-SfM'):
        download_train(args.data)
        download_test(args.data)

    # Data Loader
    log_debug("Creating dataloaders for dataset in %s", args.data)
    
    train_tf = TuplesTransform(shortest_size=data_config.getint("train_shortest_size"),
                               longest_max_size=data_config.getint("train_longest_max_size"),
                               rgb_mean=data_config.getstruct("rgb_mean"),
                               rgb_std=data_config.getstruct("rgb_std"),
                               random_flip=data_config.getboolean("random_flip"),
                               random_scale=data_config.getstruct("random_scale"))

    # Training dataloader
    train_db = TuplesDataset(root_dir=args.data,
                             name=data_config.get("training_dataset"),
                             mode='train',
                             neg_num=data_config.getint("neg_num"),
                             query_size=data_config.getint("train_query_size"),
                             pool_size=data_config.getint("train_pool_size"),
                             transform=train_tf,
                             batch_size=data_config.getint("train_batch_size"),
                             num_workers=data_config.getint("num_workers"))

    train_dl = data.DataLoader(train_db,
                               batch_sampler=None,
                               batch_size=data_config.getint("train_batch_size"),
                               collate_fn=iss_collate_fn,
                               pin_memory=True,
                               num_workers=data_config.getint("num_workers"),
                               shuffle=True,
                               drop_last=True)

    # Validation dataloader
    val_tf = TuplesTransform(shortest_size=data_config.getint("train_shortest_size"),
                             longest_max_size=data_config.getint("train_longest_max_size"),
                             rgb_mean=data_config.getstruct("rgb_mean"),
                             rgb_std=data_config.getstruct("rgb_std"),
                             random_flip=data_config.getboolean("random_flip"),
                             random_scale=data_config.getstruct("random_scale"))

    val_db = TuplesDataset(root_dir=args.data,
                           name=data_config.get("training_dataset"),
                           mode='val',
                           neg_num=data_config.getint("neg_num"),
                            #    query_size=float('inf'),
                            #    pool_size=float('inf'),
                           query_size=data_config.getint("train_query_size"),
                           pool_size=data_config.getint("train_pool_size"),
                           transform=val_tf,
                           batch_size=data_config.getint("train_batch_size"),
                           num_workers=data_config.getint("num_workers"))

    val_dl = data.DataLoader(val_db,
                            batch_sampler=None,
                            batch_size=data_config.getint("train_batch_size"),
                            collate_fn=iss_collate_fn,
                            pin_memory=True,
                            num_workers=data_config.getint("num_workers"),
                            shuffle=True,
                            drop_last=True)    
    
    return train_dl, val_dl
    

def computer_PCA_layer(model, train_dataloader, config, varargs):
    
    log_debug('Compute PCA Layer')
        
    data_config = config["dataloader"]
    global_config = config["global"]
    
    batch_size = 1

    # Set model to eval mode
    model.eval()
    model = model.cuda(varargs["device"])

    # output Dim
    inp_dim = model.ret_head.inp_dim
    out_dim = model.ret_head.out_dim

    dataset = train_dataloader.dataset
    num_samples = global_config.getint("num_samples")

    if num_samples> len(dataset.images):
        num_samples = len(dataset.images)
        
    with torch.no_grad():
            
        # Prepare query loader
        log_debug('Extracting descriptors for PCA {%s}--{%s} fo {%s}:', inp_dim, out_dim, num_samples)

        # Random 
        idxs = torch.randperm(len(dataset.images))[:num_samples]

        db_tf = ImagesTransform(shortest_size=data_config.getint("train_shortest_size"),
                                longest_max_size=data_config.getint("train_longest_max_size"),
                                rgb_mean=data_config.getstruct("rgb_mean"),
                                rgb_std=data_config.getstruct("rgb_std")
                                )
            
        db_data = ImagesFromList(root='', 
                                 images=[dataset.images[i] for i in idxs], 
                                 transform=db_tf)
            
        db_dl = torch.utils.data.DataLoader( db_data, 
                                             batch_size = batch_size, 
                                             shuffle=True,
                                             sampler=None, 
                                             num_workers=data_config.getint("num_workers"), 
                                             pin_memory=True
                                            )
        # Extract  vectors
        vecs = torch.zeros(inp_dim, len(db_data.images) ).cuda()

        for it, batch in tqdm(enumerate(db_dl), total=len(db_dl)):
  
            # Upload batch
            batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

            pred = model(**batch, do_prediction=True, do_whitening=False)
                
            # distributed.barrier()

            vecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]

            del pred
                    
    
    log_debug('Compute PCA, Takes a while')
    vecs = vecs.detach().cpu().numpy()
    
    m, P  = PCA_whitenlearn_shrinkage(vecs)
    m, P = m.T, P.T
    
    # create layer
    layer = nn.Linear(inp_dim, out_dim, bias=True)
    
    projection = torch.Tensor(P[:layer.weight.shape[0], :])
    layer.weight.data = projection.to(layer.weight.device)
    
    projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
    layer.bias.data = projected_shift[:layer.weight.shape[0]].to(layer.bias.device)
    
    return layer


def make_model(args, config, train_dataloader, **varargs):
    # parse params with default values

    body_config = config["body"]
    ir_config = config["global"]
    data_config = config["dataloader"]
    
    # To computer PCA
    img_size = data_config["train_longest_max_size"]

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config.get("arch"))

    body_fn = models.__dict__[body_config.get("arch")]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, config=body_config, **body_params)

    if body_config.getboolean("pretrained"):
        arch = body_config.get("arch")

        # vgg with bn or without
        if body_config.get("arch").startswith("vgg"):
            if body_config["normalization_mode"] != 'off':
                arch = body_config.get("arch") + '_bn'

        # Download pre trained model
        log_debug("Downloading pre - trained model weights %s", body_config.get("source_url"))

        if body_config.get("source_url") == "cvut":
            
            if body_config.get("arch") not in model_urls_cvut:
                raise ValueError(" body arch not found in cvut witch  source_url = pytorch")
            
            state_dict = load_state_dict_from_url(model_urls_cvut[arch])


            converted_model = body.convert_cvut(state_dict)

        elif body_config.get("source_url") == "pytorch":
            if body_config.get("arch") not in model_urls:
                raise ValueError(" body arch not found in pytorch ")
            
            state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
   
            converted_model = body.convert(state_dict)
      
        else:
            raise ValueError(" try source_url = cvut  or pytorch  ")


        folder = args.directory + "ImageNet"

        if not path.exists(folder):
            log_debug("Create path to save pretrained backbones: %s ", folder)
            makedirs(folder)

        body_path = folder + "/" + arch + ".pth"
        log_debug("Saving pretrained backbones in : %s ", body_path)
        torch.save(converted_model, body_path)

        # Load  converted weights to model
        body.load_state_dict(torch.load(body_path, map_location="cpu"))

        # Freeze modules in backbone
        for n, m in body.named_modules():
            for mod_id in range(1, body_config.getint("num_frozen") + 1):
                if ("mod%d" % mod_id) in n:
                    freeze_params(m)

    else:
        log_info("Initialize body to train from scratch")
        init_weights(body, body_config)

    body_dim = body.out_dim
    # OUTPUT_DIM[body_config.get("arch")]

    # Create Image Retrieval
    global_loss = globalFeatureLoss(name=ir_config.get("loss"),
                                    sigma=ir_config.getfloat("loss_margin"))
    

    global_algo = globalFeatureAlgo(loss=global_loss,
                                    batch_size=1)
    
    output_dim = ir_config.getint("out_dim")
    
    global_head = AttentionHead(    inp_dim=body_dim,
                                    out_dim=output_dim,
                                    attention=ir_config.get("attention"),
                                    num_heads=ir_config.getint("num_heads"),
                                    resolution=ir_config.getstruct("resolution"),             
                                    pooling=ir_config.getstruct("pooling"),
                                    do_withening=ir_config.getboolean("whithening"),
                                    norm_act=norm_act_dynamic)
    
    
    # Create a generic image retrieval network
    net = ImageRetrievalNet(body,
                            global_algo,
                            global_head)
   
    # Compute whithening layer or load if found from previous experiments
    if ir_config.getboolean("whithening"):
        
        # Init
        whithen_folder = path.join(args.directory, "Whithen")

        if not path.exists(whithen_folder):
            log_debug("Create path to save whithening pre computed layer: %s ", whithen_folder)
            makedirs(whithen_folder)
        
        whithen_layer_path = path.join(whithen_folder,
                                       data_config.get("training_dataset")  + "_"   + 
                                       body_config.get("arch")              + "_"   + 
                                       str(output_dim)                      + "_"   + 
                                       str(img_size)                        + ".pth")
        
        # Avoid recomputing same layer for further experiments
        if ( not path.isfile(whithen_layer_path) or ir_config.getboolean("update")):
            log_debug("Whithening not computed before or update required: %s ", whithen_layer_path)
            
            # compute layer
            whiten_layer = computer_PCA_layer(net, train_dataloader, config, varargs)
            
            # Save layer
            torch.save(whiten_layer.state_dict(), whithen_layer_path)

        # load
        log_debug("Load : %s ", whithen_layer_path)
        state_dic = torch.load(whithen_layer_path, map_location="cpu")
        
        net.ret_head.whiten.load_state_dict(state_dic)
       
    return net, output_dim


def make_optimizer(model, config, epoch_length):

    body_config = config["body"]

    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    # Base learning rate and weight decay
    LR              = optimizer_config.getfloat("lr")
    WEIGHT_DECAY    = optimizer_config.getfloat("weight_decay")

    # Tunning learning rate and weight decay
    lr_coefs            = optimizer_config.getstruct("lr_coefs")
    weight_decay_coefs  = optimizer_config.getstruct("weight_decay_coefs")

    # Separate classifier parameters from all others
    net_norm_params      = []
    net_other_params     = []
       
    # body
    for k, m in model.body.named_modules():
  
        if any(isinstance(m, layer) for layer in NORM_LAYERS):
            net_norm_params  += [p for p in m.parameters() if p.requires_grad]
            
        elif any(isinstance(m, layer) for layer in OTHER_LAYERS):
            net_other_params += [p for p in m.parameters() if p.requires_grad]
        
    # ret head
    att_params      = []
    pool_params     = []
    whiten_params   = []
        
    for k, v in model.ret_head.named_children():
   
        if k.find("attention") != -1:
            att_params += [p for p in v.parameters() if p.requires_grad]
            
        if k.find("pool") != -1:
            pool_params += [p for p in v.parameters() if p.requires_grad]
        
        elif k.find("whiten")!= -1:
            whiten_params += [p for p in v.parameters() if p.requires_grad]

    # Transformer head
    assert len(net_norm_params) + len(net_other_params) + len(pool_params) + len(whiten_params) + len(att_params) \
                            == len([p for p in model.parameters() if p.requires_grad]), \
        "Not all parameters that require grad are accounted for in the optimizer"
        
    # Set-up optimizer hyper-parameters
    parameters = [
        # body norm 
        {
            "params": net_norm_params,
            "lr":           LR              if not body_config.getboolean("bn_frozen") else 0.,
            "weight_decay": WEIGHT_DECAY    if optimizer_config.getboolean("weight_decay_norm") else 0
        },
        # body other 
        {
            "params": net_other_params,
            "lr":           LR        ,
            "weight_decay": WEIGHT_DECAY 
        },
        # pool
        {
            "params": pool_params,
            "lr":           LR * lr_coefs["pool"],
            "weight_decay": WEIGHT_DECAY * weight_decay_coefs["pool"]
        },
        # whiten
        {
            "params": whiten_params,
            "lr":           LR * lr_coefs["whiten"],
            "weight_decay": WEIGHT_DECAY * weight_decay_coefs["whiten"]
        },
        # attention 
        {
            "params": att_params,
            "lr":           LR * lr_coefs["attention"],
            "weight_decay": WEIGHT_DECAY * weight_decay_coefs["attention"]
        }
    ]
    
   
    # Select optimizer
    if optimizer_config.get("type") == 'SGD':
        optimizer = optim.SGD(parameters,
                              lr=LR,
                              weight_decay=optimizer_config.getfloat("weight_decay"),
                              nesterov=optimizer_config.getboolean("nesterov"))
    elif optimizer_config.get("type") == 'Adam':
        optimizer = optim.Adam(parameters,
                               lr=LR, 
                               weight_decay=WEIGHT_DECAY)
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    # Set scheduler
    scheduler = scheduler_from_config(scheduler_config, optimizer, epoch_length)

    assert scheduler_config.get("update_mode") in ("batch", "epoch")
    batch_update = scheduler_config.get("update_mode") == "batch"
    total_epochs = scheduler_config.getint("epochs")

    return optimizer, scheduler, parameters, batch_update, total_epochs


def train(model, config, dataloader, optimizer, scheduler, meters, **varargs):

    # Create tuples for training
    data_config = config["dataloader"]

    avg_neg_distance = dataloader.dataset.create_epoch_tuples(model, log_info, log_debug,
                                                              output_dim=varargs["output_dim"],
                                                              world_size=varargs["world_size"],
                                                              rank=varargs["rank"],
                                                              device=varargs["device"],
                                                              data_config=data_config)

    # Free Gpu cache
    torch.cuda.empty_cache()
    
    log_debug("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
    log_debug("Cached:    " + humanbytes(torch.cuda.memory_reserved()))

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
            
            vecs = torch.zeros(varargs["output_dim"], len(tuple)).cuda()
            
            # extract vectors
            for item_idx, item in enumerate(tuple):
                
                item = item.unsqueeze(0)

                pred =  model(img=item.cuda(), do_loss=True, do_whitening=True)
            
                vecs[:, item_idx:item_idx+1] = pred["ret_pred"]
            
            # compute loss
            loss = model.ret_algo.loss(vecs, target.cuda())
            loss.backward()
             
            # Accumulate
            losses["loss"] += (1/num_tuples) * loss.mean()
        
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


def validate(model, config, dataloader, **varargs):

    # create tuples for validation
    data_config = config["dataloader"]

    avg_neg_distance = dataloader.dataset.create_epoch_tuples(model, log_info, log_debug,
                                                              output_dim=varargs["output_dim"],
                                                              world_size=varargs["world_size"],
                                                              rank=varargs["rank"],
                                                              device=varargs["device"],
                                                              data_config=data_config)
    
    # Free Gpu cache
    # distributed.barrier()
    torch.cuda.empty_cache()
    
    log_debug("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
    log_debug("Cached:    " + humanbytes(torch.cuda.memory_reserved()))

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
                
                vecs = torch.zeros(varargs["output_dim"], len(tuple)).cuda()
                
                # extract vectors
                for item_idx, item in enumerate(tuple):
                    
                    item = item.unsqueeze(0)

                    pred =  model(img=item.cuda(), do_loss=True, do_whitening=True)
                
                    vecs[:, item_idx:item_idx+1] = pred["ret_pred"]
                
                # compute loss
                loss = model.ret_algo.loss(vecs, target.cuda())

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


def test(args, config, model, rank=None, world_size=None, **varargs):

    log_debug('Evaluating network on test datasets...')

    # Eval mode
    model.eval()
    data_config = config["dataloader"]

    # Average score
    avg_score = 0.0

    # Evaluate on test datasets
    list_datasets = data_config.getstruct("test_datasets")

    if data_config.get("multi_scale"):
        scales = eval(data_config.get("multi_scale"))
    else:
        scales = [1]

    for dataset in list_datasets:

        start = time.time()

        log_debug('{%s}: Loading Dataset', dataset)

        # Prepare database
        db = ParisOxfordTestDataset(root_dir=path.join(args.data, 'test', dataset),
                                    name=dataset)

        test_batch_size = 1

        with torch.no_grad():
            """ Paris and Oxford are :
                    1 - resized to a ratio of desired max size, after bbx cropping 
                    2 - normalized after that
                    3 - not flipped and not scaled (!! important for evaluation)
                
            """
            # Prepare query loader
            log_debug('{%s}: Extracting descriptors for query images', dataset)
        
            query_tf = ImagesTransform( shortest_size=data_config.getint("test_shortest_size"),
                                        longest_max_size=data_config.getint("test_longest_max_size"),
                                        rgb_mean=data_config.getstruct("rgb_mean"),
                                        rgb_std=data_config.getstruct("rgb_std")
                                        )
                
            query_data = ImagesFromList(    root='', 
                                            images=db['query_names'],
                                            bbxs= db['query_bbx'],
                                            transform=query_tf
                                            )
                
            query_dl = torch.utils.data.DataLoader( query_data, 
                                                    batch_size = test_batch_size, 
                                                    shuffle=False, 
                                                    num_workers=data_config.getint("num_workers"), 
                                                    pin_memory=True
                                                )
            
            # Extract query vectors
            qvecs = torch.zeros(varargs["output_dim"], len(query_data)).cuda()

            for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                pred = model(**batch, do_prediction=True, do_whitening=True)

                # distributed.barrier()

                qvecs[:, it * test_batch_size: (it+1) * test_batch_size] = pred["ret_pred"]

                del pred

            # Prepare negative database data loader
            log_debug('{%s}: Extracting descriptors for database images', dataset)


            database_tf = ImagesTransform(  shortest_size=data_config.getint("test_shortest_size"),
                                            longest_max_size=data_config.getint("test_longest_max_size"),
                                            rgb_mean=data_config.getstruct("rgb_mean"),
                                            rgb_std=data_config.getstruct("rgb_std")
                                            )
                
            database_data = ImagesFromList( root='', 
                                            images=db['img_names'],
                                            transform=database_tf
                                            )
                
            database_dl = torch.utils.data.DataLoader(  database_data, 
                                                        batch_size = test_batch_size, 
                                                        shuffle=False, 
                                                        num_workers=data_config.getint("num_workers"), 
                                                        pin_memory=True
                                                        )
            # Extract negative pool vectors
            database_vecs = torch.zeros(varargs["output_dim"], len(database_data)).cuda()

            for it, batch in tqdm(enumerate(database_dl), total=len(database_dl)):
                # Upload batch

                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                pred = model(**batch, do_prediction=True, do_whitening=True)

                # distributed.barrier()

                database_vecs[:, it * test_batch_size: (it+1) * test_batch_size] = pred["ret_pred"]

                del pred

        # convert to numpy
        qvecs = qvecs.cpu().numpy()
        database_vecs = database_vecs.cpu().numpy()

        # search, rank, and print
        scores = np.dot(database_vecs.T, qvecs)
        ranks  = np.argsort(-scores, axis=0)

        score = compute_map_and_print(dataset, ranks, db['gnd'], log_info)
        log_info('{%s}: Running time = %s', dataset, htime(time.time()-start))

        avg_score += 0.5 * score["mAP"]

    # As Evaluation metrics
    log_info('Average score = %s', avg_score)

    return avg_score


def save_checkpoint(state, is_best, directory):
    filename = path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def main(args):

    global min_loss

    # Initialize multi-processing
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = 0, 1

    # set device
    torch.cuda.set_device(device_id)

    # set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    
    # Load configuration
    config = make_config(args)

    # Experiment Path
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
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size)

    # Initialize model
    if body_config.getboolean("pretrained"):
        log_debug("Use pre-trained model %s", body_config.get("arch"))
    else:
        log_debug("Initialize model to train from scratch %s". body_config.get("arch"))

    # Load model
    model, output_dim = make_model(args, config, train_dataloader, 
                                      rank=rank, 
                                      world_size=world_size, 
                                      device=device)

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
    optimizer, scheduler, parameters, batch_update, total_epochs = make_optimizer(model, config, epoch_length=len(train_dataloader))
   
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
    
    print(model)
    
    # Optional: evaluation only:
    if args.eval:
        log_info("Evaluation epoch %d", start_epoch - 1)

        test(args, config, model, rank=rank, world_size=world_size,
             output_dim=output_dim,
            device=device)

        log_info("Evaluation Done ..... ")

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
                            loss_weights=optimizer_config.getstruct("loss_weights")
                            )

        # Save val snapshot (only on rank 0)
        if rank == 0 and (epoch + 1) % config["general"].getint("val_interval") == 0:
            snapshot_file = path.join(exp_dir, "model_{}.pth.tar".format(epoch))

            log_debug("Saving snapshot to %s", snapshot_file)

            meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}

            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                          body=model.body.state_dict(),
                          ret_head=model.ret_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          **meters_out_dict)

        # Run validation
        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            log_info("Validating epoch %d", epoch + 1)

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
                                    loss_weights=optimizer_config.getstruct("loss_weights")
                                    )

        # Run Test
        if (epoch + 1) % config["general"].getint("test_interval") == 0:
            log_info("Testing epoch %d", epoch + 1)

            score['test'] = test(args, config, model, rank=rank, world_size=world_size,
                                 output_dim=output_dim,
                                 device=device)

            # Update the score on the last saved snapshot
            if rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot

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