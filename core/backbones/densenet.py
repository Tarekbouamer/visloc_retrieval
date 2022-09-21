import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from .misc import DenseModule
from .util import try_index, CONV_PARAMS, BN_PARAMS, FC_PARAMS


from core.utils.misc import ABN, GlobalAvgPool2d


class DenseNet(nn.Module):
    def __init__(self,
                 structure,
                 norm_act=ABN,
                 config=None,
                 input_3x3=False,
                 growth=32,
                 theta=0.5,
                 classes=0,
                 dilation=1):
        """DenseNet
        Parameters
        ----------
        structure : list of int
            Number of layers in each of the four dense blocks of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        input_3x3 : bool
            If `True` use three `3x3` convolutions in the input module instead of a single `7x7` one.
        growth : int
            Number of channels in each layer, i.e. the "growth" factor of the DenseNet.
        theta : float
            Reduction factor for the transition blocks.
        classes : int
            If not `0` also include globalFeatures average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : int or list of int
            List of dilation factors, or `1` to ignore dilation. If the dilation factor for a module is greater than `1`
            skip the pooling in the transition block right before it.
        """
        super(DenseNet, self).__init__()
        self.structure = structure
        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")

        self.bottleneck = input_3x3

        # Initial layers
        if input_3x3:

            layers = [
                ("conv1", nn.Conv2d(3, growth * 2, 3, stride=2, padding=1, bias=False)),
                ("bn1", norm_act(growth * 2)),
                ("conv2", nn.Conv2d(growth * 2, growth * 2, 3, stride=1, padding=1, bias=False)),
                ("bn2", norm_act(growth * 2)),
                ("conv3", nn.Conv2d(growth * 2, growth * 2, 3, stride=1, padding=1, bias=False)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        else:
            layers = [
                ("conv1", nn.Conv2d(3, growth * 2, 7, stride=2, padding=3, bias=False)),
                ("bn1", norm_act(growth * 2)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        self.mod1 = nn.Sequential(OrderedDict(layers))

        in_channels = growth * 2
        for mod_id in range(4):
            d = try_index(dilation, mod_id)
            s = 2 if d == 1 and mod_id > 0 else 1

            # Create transition module
            if mod_id > 0:
                out_channels = int(in_channels * theta)
                layers = [
                    ("bn", norm_act(in_channels)),
                    ("conv", nn.Conv2d(in_channels, out_channels, 1, bias=False))
                ]
                if s == 2:
                    layers.append(("pool", nn.AvgPool2d(2, 2)))
                self.add_module("tra%d" % (mod_id + 1), nn.Sequential(OrderedDict(layers)))
                in_channels = out_channels

            # Create dense module
            mod = DenseModule(in_channels, growth, structure[mod_id], norm_act=norm_act, dilation=d)
            self.add_module("mod%d" % (mod_id + 2), mod)
            in_channels = mod.out_channels

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)

        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))
        

    def copy_layer(self, inm, outm, name_in, name_out, params):
        for param_name in params:
            outm[name_out + "." + param_name] = inm[name_in + "." + param_name]
            #input()

    def convert(self, model):
        out = dict()
        num_convs = 2

        # Initial module
        if not self.bottleneck:
            self.copy_layer(model, out, "features.conv0", "mod1.conv1", CONV_PARAMS)
            self.copy_layer(model, out, "features.norm0", "mod1.bn1", BN_PARAMS)
        else:
            raise ValueError(" Not implemented yet  ")

        # Other modules
        for mod_id, num in enumerate(self.structure):
            for block_id in range(num):
                for conv_id in range(num_convs):
                    self.copy_layer(model, out,
                               "features.denseblock{}.denselayer{}.conv.{}".format(mod_id + 1, block_id + 1, conv_id + 1),
                               "mod{}.convs{}.{}.conv".format(mod_id + 2, conv_id + 1, block_id),
                               CONV_PARAMS)
                    self.copy_layer(model, out,
                               "features.denseblock{}.denselayer{}.norm.{}".format(mod_id + 1, block_id +1 , conv_id + 1),
                               "mod{}.convs{}.{}.bn".format(mod_id + 2, conv_id + 1, block_id),
                               BN_PARAMS)

                # Try copying projection module
                try:
                    self.copy_layer(model, out,
                               "features.layer{}.{}.downsample.0".format(mod_id + 1, block_id + 1, conv_id + 1),
                               "mod{}.convs{}.{}.proj_conv".format(mod_id + 2, conv_id + 1, block_id),
                               CONV_PARAMS)
                    self.copy_layer(model, out,
                               "features.layer{}.{}.downsample.1".format(mod_id + 1, block_id + 1, conv_id + 1),
                               "mod{}.convs{}.{}.proj_bn".format(mod_id + 2, conv_id + 1, block_id),
                               BN_PARAMS)
                except KeyError:
                    pass

            # Pass transitions modules
            try:
                self.copy_layer(model, out,
                                "features.transition{}.conv".format(mod_id + 1),
                                "tra{}.conv".format(mod_id + 2),
                                CONV_PARAMS)
                self.copy_layer(model, out,
                                "features.transition{}.norm".format(mod_id + 1),
                                "tra{}.bn".format(mod_id + 2),
                                BN_PARAMS)
            except KeyError:
                pass

        # Pooling and predictor
        self.copy_layer(model, out, "features.norm5", "bn_out", BN_PARAMS)

        if hasattr(self, "classifier"):
            self.copy_layer(model, out, "classifier", "classifier.fc", FC_PARAMS)

        return out

    def forward(self, x):
        x = self.mod1(x)
        x = self.mod2(x)
        x = self.tra2(x)
        x = self.mod3(x)
        x = self.tra3(x)
        x = self.mod4(x)
        x = self.tra4(x)
        x = self.mod5(x)
        x = self.bn_out(x)

        if hasattr(self, "classifier"):
            x = self.classifier(x)
        return x


_NETS = {
    "121": {"structure": [6, 12, 24, 16], "growth": 32},
    "161": {"structure": [6, 12, 36, 24], "growth": 48},
    "169": {"structure": [6, 12, 32, 32], "growth": 32},
    "201": {"structure": [6, 12, 48, 32], "growth": 32},
    "264": {"structure": [6, 12, 64, 48], "growth": 32},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "densenet" + name
    setattr(sys.modules[__name__], net_name, partial(DenseNet, **params))
    __all__.append(net_name)