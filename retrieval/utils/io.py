from os import makedirs, path


def create_folder(dir, logger=None):
    if not path.exists(dir):
        if logger: 
            logger("Create experiment path  from %s", dir)
        makedirs(dir)


def create_experiment_file(dir, extension="", logger=None):
    
    dir = path.join(dir, extension)
    
    create_folder(dir)
    
    return dir

def create_experiment_file_from_cfg(cfg, directory, logger=None):
    
    # Create export dir name if it doesnt exist in your experiment folder
    extension = "{}".format(cfg["dataloader"].get("dataset"))
    extension += "_{}".format(cfg["body"].get("arch"))

    extension += "_{}_m{:.2f}".format(cfg["global"].get("loss"),
                                      cfg["global"].getfloat("loss_margin"))

    if cfg["global"].getstruct("pooling"):
        extension += "_{}".format(cfg["global"].getstruct("pooling")["name"])
        
    
    if cfg["global"].get("attention"):
        extension += "_{}_encs{}_h{}".format(cfg["global"].get("attention"),
                                            cfg["global"].getfloat("num_encs"),
                                            cfg["global"].getfloat("num_heads"))

    extension += "_{}_lr{:.1e}_wd{:.1e}".format(cfg["optimizer"].get("type"),
                                                cfg["optimizer"].getfloat("lr"),
                                                cfg["optimizer"].getfloat("weight_decay"))

    extension += "_nnum{}".format(cfg["dataloader"].getint("neg_num"))
    
    extension += "_bsize{}_imsize{}".format(cfg["dataloader"].getint("batch_size"),
                                            cfg["dataloader"].getint("max_size"))

    create_experiment_file(directory, extension)


    return directory
  

def create_withen_file_from_cfg(cfg, directory,logger=None):
    
    out_dim = cfg["global"].get("global_dim")

    DATASET = cfg["dataloader"].get("dataset")
    ARCH = cfg["body"].get("arch")
    LEVELS = str(len(cfg["body"].getstruct("features_scales")))
    DIM = str(out_dim)
    IM_SIZE = str(cfg["dataloader"].getint("max_size"))
    
    whithen_path = path.join(directory,       
                              DATASET          + "_"   + 
                              ARCH                          + "_"   + 
                              "L"    +      LEVELS          + "_"   + 
                              "D"    +      DIM             + "_"   + 
                              "Size" +    IM_SIZE           + 
                              ".pth"
                            )
    return whithen_path