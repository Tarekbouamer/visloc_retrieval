from __future__ import absolute_import, division, print_function

import copy
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.registry.factory import load_pretrained
from core.registry.register import get_pretrained_cfg

from retrieval.models.base import RetrievalBase

from .misc import _cfg, register_retrieval

sys.path.append('../')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return self.__class__.__name__


class ConcatTable(nn.Module):
    '''ConcatTable container in Torch7.
    '''

    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        return [self.layer1(x), self.layer2(x)]


class Identity(nn.Module):
    '''
    nn.Identity in Torch7.
    '''

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (skip connection)'


class Reshape(nn.Module):
    '''
    nn.Reshape in Torch7.
    '''

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def __repr__(self):
        return self.__class__.__name__ + ' (reshape to size: {})'.format(" ".join(str(x) for x in self.shape))


class CMul(nn.Module):
    '''
    nn.CMul in Torch7.
    '''

    def __init__(self):
        super(CMul, self).__init__()

    def forward(self, x):
        return x[0]*x[1]

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
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''

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


def __unfreeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = True


def __freeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = False


def __print_freeze_status__(model):
    '''print freeze stagus. only for debugging purpose.
    '''
    for i, module in enumerate(model.named_children()):
        for param in module[1].parameters():
            print('{}:{}'.format(module[0], str(param.requires_grad)))


def __load_weights_from__(module_dict, load_dict, modulenames):
    for modulename in modulenames:
        module = module_dict[modulename]
        print('loaded weights from module "{}" ...'.format(modulename))
        module.load_state_dict(load_dict[modulename])


def __deep_copy_module__(module, exclude=[]):
    modules = {}
    for name, m in module.named_children():
        if name not in exclude:
            modules[name] = copy.deepcopy(m)
            print('deep copied weights from layer "{}" ...'.format(name))
    return modules


def __cuda__(model):
    if torch.cuda.is_available():
        model.cuda()
    return model


class Delf(RetrievalBase):
    def __init__(
            self,
            cfg=None,
            ncls=None,
            load_from=None,
            arch='resnet50',
            stage='inference',
            target_layer='layer3',
            use_random_gamma_rescale=False):
        super(Delf, self).__init__(cfg)

        self.arch = arch
        self.stage = stage
        self.target_layer = target_layer
        self.load_from = load_from
        self.use_random_gamma_rescale = use_random_gamma_rescale

        self.module_list = nn.ModuleList()
        self.module_dict = {}
        self.end_points = {}

        in_c = self.__get_in_c__()
        if self.stage in ['finetune']:
            use_pretrained_base = True
            exclude = ['avgpool', 'fc']

        elif self.stage in ['keypoint']:
            use_pretrained_base = False
            self.use_l2_normalized_feature = True
            if self.target_layer in ['layer3']:
                exclude = ['layer4', 'avgpool', 'fc']
            if self.target_layer in ['layer4']:
                exclude = ['avgpool', 'fc']

        else:
            assert self.stage in ['inference']
            use_pretrained_base = False
            self.use_l2_normalized_feature = True
            if self.target_layer in ['layer3']:
                exclude = ['layer4', 'avgpool', 'fc']
            if self.target_layer in ['layer4']:
                exclude = ['avgpool', 'fc']

        if self.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print('[{}] loading {} pretrained ImageNet weights ... It may take few seconds...'
                  .format(self.stage, self.arch))
            module = models.__dict__[self.arch](pretrained=use_pretrained_base)
            module_state_dict = __deep_copy_module__(module, exclude=exclude)
            module = None

            # endpoint: base
            submodules = []
            submodules.append(module_state_dict['conv1'])
            submodules.append(module_state_dict['bn1'])
            submodules.append(module_state_dict['relu'])
            submodules.append(module_state_dict['maxpool'])
            submodules.append(module_state_dict['layer1'])
            submodules.append(module_state_dict['layer2'])
            submodules.append(module_state_dict['layer3'])
            self.__register_module__('base', submodules)

            # build structure.
            if self.stage in ['finetune']:
                # endpoint: layer4, pool
                self.__register_module__('layer4', module_state_dict['layer4'])
                self.__register_module__('pool', nn.AvgPool2d(
                    kernel_size=7, stride=1, padding=0,
                    ceil_mode=False, count_include_pad=True))
            elif self.stage in ['keypoint', 'inference']:
                # endpoint: attn, pool
                self.__register_module__(
                    'attn', SpatialAttention2d(in_c=in_c, act_fn='relu'))
                self.__register_module__('pool', WeightedSum2d())

            if self.stage not in ['inference']:
                # endpoint: logit
                submodules = []
                submodules.append(nn.Conv2d(in_c, ncls, 1))
                submodules.append(Flatten())
                self.__register_module__('logits', submodules)

            # load weights.
            if self.stage in ['keypoint']:
                load_dict = torch.load(self.load_from)
                __load_weights_from__(
                    self.module_dict, load_dict, modulenames=['base'])
                __freeze_weights__(self.module_dict, freeze=['base'])
                print('load model from "{}"'.format(load_from))
            elif self.stage in ['inference']:
                # load_dict = torch.load(self.load_from)
                # __load_weights_from__(self.module_dict, load_dict, modulenames=[
                #                       'base', 'attn', 'pool'])
                print('load model from "{}"'.format(load_from))

    def __register_module__(self, modulename, module):
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        self.module_list.append(module)
        self.module_dict[modulename] = module

    def __get_in_c__(self):
        # adjust input channels according to arch.
        if self.arch in ['resnet18', 'resnet34']:
            in_c = 512
        elif self.arch in ['resnet50', 'resnet101', 'resnet152']:
            if self.stage in ['finetune']:
                in_c = 2048
            elif self.stage in ['keypoint', 'inference']:
                if self.target_layer in ['layer3']:
                    in_c = 1024
                elif self.target_layer in ['layer4']:
                    in_c = 2048
        return in_c

    def __forward_and_save__(self, x, modulename):
        module = self.module_dict[modulename]
        x = module(x)
        self.end_points[modulename] = x
        return x

    def __forward_and_save_feature__(self, x, model, name):
        x = model(x)
        self.end_points[name] = x.data
        return x

    def __gamma_rescale__(self, x, min_scale=0.3535, max_scale=1.0):
        '''max_scale > 1.0 may cause training failure.
        '''
        h, w = x.size(2), x.size(3)
        assert w == h, 'input must be square image.'
        gamma = random.uniform(min_scale, max_scale)
        new_h, new_w = int(h*gamma), int(w*gamma)
        x = F.upsample(x, size=(new_h, new_w), mode='bilinear')
        return x

    def get_endpoints(self):
        return self.end_points

    def get_feature_at(self, modulename):
        return copy.deepcopy(self.end_points[modulename].data.cpu())

    def write_to(self, state):
        if self.stage in ['finetune']:
            state['base'] = self.module_dict['base'].state_dict()
            state['layer4'] = self.module_dict['layer4'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        elif self.stage in ['keypoint']:
            state['base'] = self.module_dict['base'].state_dict()
            state['attn'] = self.module_dict['attn'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        else:
            assert self.stage in ['inference']
            raise ValueError('inference does not support model saving!')

    def forward_for_serving(self, x):
        '''
        This function directly returns attention score and raw features
        without saving to endpoint dict.
        '''
        x = self.__forward_and_save__(x, 'base')
        if self.target_layer in ['layer4']:
            x = self.__forward_and_save__(x, 'layer4')
        ret_x = x
        if self.use_l2_normalized_feature:
            F.normalize(x, p=2, dim=1)
        else:
            pass
        attn_score = self.__forward_and_save__(x, 'attn')
        ret_s = attn_score
        return ret_x.data.cpu(), ret_s.data.cpu()

    def forward(self, x):
        if self.stage in ['finetune']:
            x = self.__forward_and_save__(x, 'base')
            x = self.__forward_and_save__(x, 'layer4')
            x = self.__forward_and_save__(x, 'pool')
            x = self.__forward_and_save__(x, 'logits')
        elif self.stage in ['keypoint']:
            if self.use_random_gamma_rescale:
                x = self.__gamma_rescale__(x)
            x = self.__forward_and_save__(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.__forward_and_save__(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            else:
                attn_x = x
            attn_score = self.__forward_and_save__(x, 'attn')
            x = self.__forward_and_save__([attn_x, attn_score], 'pool')
            x = self.__forward_and_save__(x, 'logits')

        elif self.stage in ['inference']:
            x = self.__forward_and_save__(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.__forward_and_save__(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            else:
                attn_x = x
            attn_score = self.__forward_and_save__(x, 'attn')
            x = self.__forward_and_save__([attn_x, attn_score], 'pool')

        else:
            raise ValueError(
                'unsupported stage parameter: {}'.format(self.stage))
        
        x= x.view(x.size(0), -1)
        return {'features': x}

    @torch.no_grad()
    def extract(self, data):
        """Extract features from an image"""
        return self.forward(data["image"])


default_cfgs = {
    'delf':
        _cfg(drive='https://drive.google.com/uc?id=1jCBCslDga4VyuLVfynes7sov2hEfq6j_',
             backbone="resnet50", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
}


def _create_model(name, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """ create a model """

    # default cfg
    model_cfg = get_pretrained_cfg(name)
    cfg = {**model_cfg, **cfg}

    # create model
    model = Delf(cfg=cfg)

    # load pretrained weights
    if pretrained:
        model_path = "/home/loc/3D/visloc_retrieval/hub/pretrained_model/ldmk/model/keypoint/ckpt/fix.pth.tar"
        load_dict = torch.load(model_path)
        print(load_dict.keys())
        print()

        # model.base.load_state_dict(load_dict["base"])
        # model.attn.load_state_dict(load_dict["attn"])
        # model.pool.load_state_dict(load_dict["pool"])

        model.module_dict["base"].load_state_dict(load_dict["base"])
        model.module_dict["attn"].load_state_dict(load_dict["attn"])
        model.module_dict["pool"].load_state_dict(load_dict["pool"])


        # load_pretrained(model, name, cfg, state_key="model")

    return model


@register_retrieval
def delf(cfg=None, pretrained=True, **kwargs):
    """Constructs  a Delf model"""
    return _create_model('delf', cfg, pretrained, **kwargs)