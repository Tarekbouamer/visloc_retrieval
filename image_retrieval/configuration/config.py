import ast
import configparser
from os import path, listdir
import io

_DEFAULTS_DIR = path.abspath(path.join(path.split(__file__)[0], "defaults"))
DEFAULTS = dict()

for file in listdir(_DEFAULTS_DIR):
    name, ext = path.splitext(file)
    if ext == ".ini":
        DEFAULTS[name] = path.join(_DEFAULTS_DIR, file)
