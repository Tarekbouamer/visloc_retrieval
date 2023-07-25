import atexit
import functools
import logging
import os
import sys
from math import log10

import torch
from loguru import logger
from tabulate import tabulate
from termcolor import colored

# from detectron2.utils.file_io import PathManager
from .meters import AverageMeter


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


@functools.lru_cache()
def setup_logger(output=None, *, color=True, name="visloc", abbrev_name=None, suffix=None):
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

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M")
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
        if suffix is not None:
            filename = os.path.join(output, suffix + ".txt")
        else:
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


# def get_logger():
#     return logging.getLogger(_NAME)


def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


def iteration(logger, summary, phase, global_step, epoch, num_epochs, step, num_steps, values, multiple_lines=False):

    # Build message and write summary
    msg = _current_total_formatter(
        epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)
    for k, v in values.items():
        if isinstance(v, AverageMeter):
            msg += "\n" if multiple_lines else "" + \
                "\t{}={:.3f} ({:.3f})".format(k, v.value.item(), v.mean.item())
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k),
                                   v.value.item(), global_step)
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


def init_loguru(name="log", app_name="VisLoc", log_file=None, file_name="logging"):

    # if log_file is directory then create a file name
    if os.path.isdir(log_file):
        log_file = os.path.join(log_file, file_name + ".log")

    logger_format = (
        "<g>{time:YYYY-MM-DD HH:mm}</g>|"
        f"<m>{app_name}</m>|"
        "<level>{level: <8}</level>|"
        "<c>{name}</c>:<c>{function}</c>:<c>{line}</c>|"
        "{extra[ip]} {extra[user]} <level>{message}</level>")

    # ip and user
    logger.configure(extra={"ip": "", "user": ""})  # Default values

    # Remove the default logger configuration
    logger.remove()
    logger.add(log_file, enqueue=True)  # Add a file sink for logging

    # You can add additional sinks for logging, such as console output
    logger.add(sys.stderr, format=logger_format, colorize=True)

    logger.success("init logger")

    return logger


def create_small_table(small_dict, fmt=".2f"):
    """
    """
    keys, values = tuple(zip(*small_dict.items()))

    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=fmt,
        stralign="center",
        numalign="center",
    )
    return table
