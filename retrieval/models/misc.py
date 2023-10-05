from core.registry.factory import create_model #
from core.registry.register import register_model

create_retrieval = create_model
register_retrieval = register_model

def _cfg(url='', drive='', out_dim=1024, **kwargs):
    return {
        'url': url,
        'drive': drive,
        'reduction': False,
        'input_size': (3, 1024, 1024),
        'out_dim': out_dim,
        **kwargs
    }