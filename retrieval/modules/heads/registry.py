import sys
from collections import defaultdict

# dict of sets to check membership of head in module
_module_to_heads = defaultdict(set)
_head_to_module = {}                    # mapping of head names to module names
_head_entrypoints = {}                  # mapping of head names to entrypoint fns


def is_head(head_name):
    """ Check if a head name exists
    """
    return head_name in _head_entrypoints


def head_entrypoint(head_name):
    """Fetch a head entrypoint for specified head name
    """
    return _head_entrypoints[head_name]


def register_head(fn):

    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add head to __all__ in module
    head_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(head_name)
    else:
        mod.__all__ = [head_name]

    # add entries to registry dict/sets
    _head_entrypoints[head_name] = fn
    _head_to_module[head_name] = module_name
    _module_to_heads[module_name].add(head_name)

    return fn
