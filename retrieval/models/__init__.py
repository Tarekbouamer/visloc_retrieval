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
    sfm_resnet50_c4_how_128
)

from .netvlad import (
    vgg16_netvlad
)

from .patch_netvlad import (
    mapillary_vgg16_patchnetvlad_128,
    mapillary_vgg16_patchnetvlad_512,
    mapillary_vgg16_patchnetvlad_4096,
)

from .delf import (
    delf
)
__all__ = [
    'create_retrieval',
    'register_retrieval',
    'gl18_resnet50_gem_2048',
    'sfm_resnet50_c4_gem_1024',
    'sfm_resnet50_gem_2048',
    'sfm_resnet101_c4_gem_1024',
    'sfm_resnet101_gem_2048',
    'sfm_resnet18_how_128',
    'sfm_resnet50_c4_how_128',
    'vgg16_netvlad',
    'mapillary_vgg16_patchnetvlad_128',
    'mapillary_vgg16_patchnetvlad_512',
    'mapillary_vgg16_patchnetvlad_4096'
    'delf'
]
