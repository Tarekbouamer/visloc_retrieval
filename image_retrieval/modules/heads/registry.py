import sys

from collections import defaultdict

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_pretrained_cfg_key', 'has_pretrained_cfg_key', 'get_pretrained_cfg_value', 'is_model_pretrained']


_module_to_models = defaultdict(set)    # dict of sets to check membership of model in module
_model_to_module = {}                   # mapping of model names to module names
_model_entrypoints = {}                 # mapping of model names to entrypoint fns
# _model_has_pretrained = set()           # set of model names that have pretrained weight url present
# _model_pretrained_cfgs = dict()         # central repo for model default_cfgs


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]
  

def register_head(fn):
    
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    
    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
       
    return fn