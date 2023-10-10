from .misc import (
    create_retrieval,  # noqa: F401
    register_retrieval,  # noqa: F401
)
from .resnet_gem import (
    gl18_resnet50_gem_2048,
    sfm_resnet50_c4_gem_1024,
    sfm_resnet50_gem_2048,
    sfm_resnet101_c4_gem_1024,
    sfm_resnet101_gem_2048,
)
from .resnet_how import (
    sfm_resnet18_how_128,
    sfm_resnet101_c4_how_128,
)

__all__ = [
    'create_retrieval',
    'register_retrieval',
    'sfm_resnet101_c4_gem_1024',
    'sfm_resnet101_gem_2048',
    'sfm_resnet50_gem_2048',
    'sfm_resnet50_c4_gem_1024',
    'gl18_resnet50_gem_2048',
    'sfm_resnet18_how_128',
    'sfm_resnet101_c4_how_128',
]
