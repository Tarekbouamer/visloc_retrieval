from argparse import ArgumentError
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
    extension = "{}".format(cfg.dataloader.dataset)
    extension += "_{}".format(cfg.body.name)
    extension += "_{}_m{:.2f}".format(cfg.loss.type,
                                      cfg.loss.margin)
    extension += "_{}_lr{:.1e}_wd{:.1e}".format(cfg.optimizer.type,
                                                cfg.optimizer.lr,
                                                cfg.optimizer.weight_decay)
    extension += "_nnum{}".format(cfg.dataloader.neg_num)
    extension += "_bsize{}_imsize{}".format(cfg.dataloader.batch_size,
                                            cfg.dataloader.max_size)

    # Create export dir
    create_experiment_file(directory, extension)

    return directory


def create_withen_file_from_cfg(cfg, directory, logger=None):

    DATASET = cfg.dataloader.dataset
    ARCH = cfg.body.name
    LEVELS = str(len(cfg.body.features_scales))
    DIM = str(cfg.body.out_dim)
    IM_SIZE = str(cfg.dataloader.max_size)

    whithen_path = path.join(directory,
                             DATASET + "_" +
                             ARCH + "_" +
                             "L" + LEVELS + "_" +
                             "D" + DIM + "_" +
                             "Size" + IM_SIZE +
                             ".pth"
                             )
    return whithen_path


def csv_float(seq, sep=','):
    ''' Convert a string of comma separated values to floats
        @returns iterable of floats
    '''
    values = []
    for v0 in seq.split(sep):
        try:
            v = float(v0)
            values.append(v)
        except ValueError:
            raise ArgumentError(
                'Invalid value %s, values must be a number' % v)
    return values
