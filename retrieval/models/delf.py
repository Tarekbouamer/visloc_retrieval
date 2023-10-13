# from __future__ import absolute_import, division, print_function

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.registry.register import get_pretrained_cfg

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval

# def __get_in_c__(self):
#     # adjust input channels according to arch.
#     if self.arch in ['resnet18', 'resnet34']:
#         in_c = 512
#     elif self.arch in ['resnet50', 'resnet101', 'resnet152']:
#         if self.stage in ['finetune']:
#             in_c = 2048
#         elif self.stage in ['keypoint', 'inference']:
#             if self.target_layer in ['layer3']:
#                 in_c = 1024
#             elif self.target_layer in ['layer4']:
#                 in_c = 2048
#     return in_c


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return self.__class__.__name__


class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()

    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
            'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        # element-wise multiplication
        y = x * weights
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1

    def __repr__(self):
        return self.__class__.__name__


class SpatialAttention2d(nn.Module):
    """SpatialAttention2d
        2-layer 1x1 conv network with softplus activation.
        <!!!> attention score normalization will be added for experiment.
    """

    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)                 # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)                    # 1x1 conv
        # use default setting.
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


def __deep_copy_module__(module, exclude=[]):
    modules = {}
    for name, m in module.named_children():
        if name not in exclude:
            modules[name] = copy.deepcopy(m)
            print('deep copied weights from layer "{}" ...'.format(name))
    return modules


class Delf(RetrievalBase):
    def __init__(
            self,
            cfg=None,
            num_classes=586,
            target_layer='layer3',
            stage='finetune'):
        super(Delf, self).__init__(cfg)

        # stage
        self.stage = stage

        self.target_layer = target_layer

        self.module_list = nn.ModuleList()
        self.module_dict = {}
        self.end_points = {}

        if self.stage in ['finetune']:
            use_pretrained_base = True
            exclude = ['avgpool', 'fc']

        if self.stage in ['inference']:
            use_pretrained_base = False
            self.use_l2_normalized_feature = True

            if self.target_layer in ['layer3']:
                exclude = ['layer4', 'avgpool', 'fc']
            if self.target_layer in ['layer4']:
                exclude = ['avgpool', 'fc']

        # base module
        module = models.__dict__[self.cfg.backbone](
            pretrained=use_pretrained_base)
        module_state_dict = __deep_copy_module__(module, exclude=exclude)
        module = None

        submodules = []
        submodules.append(module_state_dict['conv1'])
        submodules.append(module_state_dict['bn1'])
        submodules.append(module_state_dict['relu'])
        submodules.append(module_state_dict['maxpool'])
        submodules.append(module_state_dict['layer1'])
        submodules.append(module_state_dict['layer2'])
        submodules.append(module_state_dict['layer3'])
        self.register_module('base', submodules)

        # layer 4
        if self.stage == "finetune":
            self.register_module('layer4', module_state_dict['layer4'])
            self.register_module('pool', nn.AvgPool2d(
                kernel_size=7, stride=1, padding=0,
                ceil_mode=False, count_include_pad=True))

        # attn + pool
        if self.stage == "inference":
            self.register_module('attn', SpatialAttention2d(in_c=self.cfg.out_dim, act_fn='relu'))
            self.register_module('pool', WeightedSum2d())

        # logits k
        if self.stage  in ['finetune']:
            submodules = []
            submodules.append(nn.Conv2d(self.cfg.out_dim, num_classes, 1))
            submodules.append(Flatten())
            self.register_module('logits', submodules)


    def register_module(self, modulename, module):
        """register module to module_list and module_dict"""
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        self.module_list.append(module)
        self.module_dict[modulename] = module

    def forward_and_save(self, x, modulename):
        module = self.module_dict[modulename]
        x = module(x)
        self.end_points[modulename] = x
        return x

    def forward(self, x):
        """Forward pass"""
        
        # fine-tune
        if self.stage in ['finetune']:
            x = self.forward_and_save(x, 'base')
            x = self.forward_and_save(x, 'layer4')
            x = self.forward_and_save(x, 'pool')
            x = self.forward_and_save(x, 'logits')
        
        elif self.stage in ['inference']:
            x = self.forward_and_save(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.forward_and_save(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            attn_score = self.forward_and_save(x, 'attn')
            x = self.forward_and_save([attn_x, attn_score], 'pool')

        return {'features': x.view(x.size(0), -1)}

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data["image"])


default_cfgs = {
    'delf_finetune':
        _cfg(drive='https://drive.google.com/uc?id=1jCBCslDga4VyuLVfynes7sov2hEfq6j_',
             backbone="resnet50", out_dim=2048, num_classes=586, target_layer='layer4', stage='finetune'),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = Delf(cfg=cfg)
    print(model)

    # load pretrained weights
    if pretrained:
        model_path = "/home/loc/3D/visloc_retrieval/hub/pretrained_model/ldmk/model/finetune/ckpt/bestshot.pth.tar"
        load_dict = torch.load(model_path)
        print(load_dict.keys())
        print()
        input()

        # model.base.load_state_dict(load_dict["base"])
        # model.attn.load_state_dict(load_dict["attn"])
        # model.pool.load_state_dict(load_dict["pool"])

        # 
        model.module_dict["base"].load_state_dict(load_dict["base"])
        
        if "layer4" in model.module_dict.keys():
            model.module_dict["layer4"].load_state_dict(load_dict["layer4"])
        if "attn" in model.module_dict.keys():
            model.module_dict["attn"].load_state_dict(load_dict["attn"])
        if "pool" in model.module_dict.keys():
            model.module_dict["pool"].load_state_dict(load_dict["pool"])
        if "logits" in model.module_dict.keys():
            model.module_dict["logits"].load_state_dict(load_dict["logits"])

        # load_pretrained(model, name, cfg, state_key="model")

    return model


@register_retrieval
def delf_finetune(cfg=None, pretrained=True, **kwargs):
    """Constructs  a Delf model"""
    return _create_model('delf_finetune', cfg, pretrained, **kwargs)
