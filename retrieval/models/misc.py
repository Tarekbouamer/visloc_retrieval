from core.registry.factory import create_model #
from core.registry.register import register_model

create_retrieval = create_model
register_retrieval = register_model

def _cfg(url='', drive='', **kwargs):
    return {
        'url': url,
        'drive': drive,
        **kwargs
    }