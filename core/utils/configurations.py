import ast
import configparser
from os import path, listdir
import io

_CONVERTERS = {
    "struct": ast.literal_eval
}



def load_config(config_file, defaults_file=None):
    parser = configparser.ConfigParser(allow_no_value=True, converters=_CONVERTERS)
    if defaults_file:
        parser.read([defaults_file, config_file])
    else:
        parser.read([config_file])
    return parser


def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str
  
  
def make_config(config_path, defauls, logger=None):
    
    if logger:
        logger.info("Loading configuration from %s", config_path)
      
    config = load_config(config_path, defauls)

    if logger:
        logger.info("\n%s", config_to_string(config))

    return config