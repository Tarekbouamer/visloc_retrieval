from .cosplace import (
    cosplace_resnet18_gem_512,
    cosplace_resnet50_gem_2048,
    cosplace_resnet101_gem_2048,
    cosplace_resnet152_gem_2048,
    cosplace_vgg16_gem_512,
    eigenplace_resnet50_gem_2048
)

from .netvlad import vgg16_netvlad
from .patch_netvlad import (
    mapillary_vgg16_patchnetvlad_128,
    mapillary_vgg16_patchnetvlad_512,
    mapillary_vgg16_patchnetvlad_4096,
)
from .resnet_gem import (
    gl18_resnet50_gem_2048,
    sfm_resnet50_c4_gem_1024,
    sfm_resnet50_gem_2048,
    sfm_resnet101_c4_gem_1024,
    sfm_resnet101_gem_2048,
)
from .resnet_how import sfm_resnet18_how_128, sfm_resnet50_c4_how_128

from .pytorch_gem import (
    sfm_resnet50_gem,
    sfm_resnet101_gem,
    sfm_resnet152_gem,
    gl18_resnet50_gem,
    gl18_resnet101_gem,
    gl18_resnet152_gem,
)


__all__ = [
    "create_retrieval",
    "register_retrieval",
    "vgg16_netvlad",
    "gl18_resnet50_gem_2048",
    "sfm_resnet50_c4_gem_1024",
    "sfm_resnet50_gem_2048",
    "sfm_resnet101_c4_gem_1024",
    "sfm_resnet101_gem_2048",
    "sfm_resnet18_how_128",
    "sfm_resnet50_c4_how_128",
    "mapillary_vgg16_patchnetvlad_128",
    "mapillary_vgg16_patchnetvlad_512",
    "mapillary_vgg16_patchnetvlad_4096",
    "cosplace_resnet50_gem_2048",
    "cosplace_resnet101_gem_2048",
    "cosplace_vgg16_gem_512",
    "cosplace_resnet18_gem_512",
    "cosplace_resnet152_gem_2048",
    "eigenplace_resnet50_gem_2048",
    "sfm_resnet50_gem",
    "sfm_resnet101_gem",
    "sfm_resnet152_gem",
    "gl18_resnet50_gem",
    "gl18_resnet101_gem",
    "gl18_resnet152_gem",
    
]
