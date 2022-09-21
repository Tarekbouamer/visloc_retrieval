import logging
from math import log10
from os import path

import atexit
import functools
import logging
import os
import sys
import time
from collections import Counter
import torch
from tabulate import tabulate
from termcolor import colored

# from detectron2.utils.file_io import PathManager

from .meters import AverageMeter

# _NAME = "Image Retrieval"
# def _current_total_formatter(current, total):
#     width = int(log10(total)) + 1
#     return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)
# def init(log_dir, name):
#     logger = logging.getLogger(_NAME)
#     logger.setLevel(logging.DEBUG)
#     # Set console logging
#     console_handler = logging.StreamHandler()
#     console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
#     console_handler.setFormatter(console_formatter)
#     console_handler.setLevel(logging.DEBUG)
#     logger.addHandler(console_handler)
#     # Setup file logging
#     file_handler = logging.FileHandler(path.join(log_dir, name + ".log"), mode="w")
#     file_formatter = logging.Formatter(fmt="%(levelname).1s %(asctime)s %(message)s", datefmt="%y-%m-%d %H:%M:%S")
#     file_handler.setFormatter(file_formatter)
#     file_handler.setLevel(logging.INFO)
#     logger.addHandler(file_handler)
# def get_logger():
#     return logging.getLogger(_NAME)
# def iteration(summary, phase, global_step, epoch, num_epochs, step, num_steps, values, multiple_lines=False):
#     logger = get_logger()
#     # Build message and write summary
#     msg = _current_total_formatter(epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)
#     for k, v in values.items():
#         if isinstance(v, AverageMeter):
#             msg += "\n" if multiple_lines else "" + "\t{}={:.3f} ({:.3f})".format(k, v.value.item(), v.mean.item())
#             if summary is not None:
#                 summary.add_scalar("{}/{}".format(phase, k), v.value.item(), global_step)
#         else:
#             msg += "\n" if multiple_lines else "" + "\t{}={:.3f}".format(k, v)
#             if summary is not None:
#                 summary.add_scalar("{}/{}".format(phase, k), v, global_step)

#     # Write log
#     logger.info(msg)

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers

def setup_logger(output=None, *, color=True, name="visloc", abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Formaters 
     
    plain_formatter = logging.Formatter("[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M")
    color_formatter = _ColorfulFormatter(
                        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                        datefmt="%m/%d %H:%M",
                        root_name=name,
                        abbrev_name=str(abbrev_name),)

    # console logging
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(color_formatter if color else plain_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # file logging
    if output is not None:
        
        filename = os.path.join(output, "log.txt")
        
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = os.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return 

# def get_logger():
#     return logging.getLogger(_NAME)

def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


def iteration(logger, summary, phase, global_step, epoch, num_epochs, step, num_steps, values, multiple_lines=False):
    
    # Build message and write summary
    msg = _current_total_formatter(epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)
    for k, v in values.items():
        if isinstance(v, AverageMeter):
            msg += "\n" if multiple_lines else "" + "\t{}={:.3f} ({:.3f})".format(k, v.value.item(), v.mean.item())
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v.value.item(), global_step)
        else:
            msg += "\n" if multiple_lines else "" + "\t{}={:.3f}".format(k, v)
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v, global_step)

    # Write log
    logger.info(msg)


def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different detectron2 components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once("detectron2." + identifier)