from typing import Optional, Any, Tuple, Union, List
import numbers
import io
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

from . import scheduler as lr_scheduler


class Empty(Exception):
    """Exception to facilitate handling of empty predictions, annotations etc."""
    pass


class GlobalAvgPool2d(nn.Module):
    
    """Global average pooling over the input's spatial dimensions"""

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class Interpolate(nn.Module):
    """nn.Module wrapper to nn.functional.interpolate"""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class ALN(nn.Module):
    """Activated Layer Normalization

    """

    _version = 2
    __constants__ = [
        "normalized_shape",
        "eps",
        "affine",
        "activation",
        "activation_param",
    ]
    normalized_shape: Tuple[int, ...]
    eps: float
    affine: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        normalized_shape:  Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        affine: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ALN, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine
        
        self.activation = activation
        self.activation_param = activation_param
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(self.normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
          
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = functional.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
            )

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(
                x, negative_slope=self.activation_param, inplace=True
            )
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def extra_repr(self):
        rep = "{normalized_shape}, eps={eps}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)
    
    
class ABN(nn.Module):
    """Activated Batch Normalization
    This gathers a BatchNorm and an activation function in a single module
    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_features",
        "affine",
        "activation",
        "activation_param",
    ]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _get_momentum_and_training(self):
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = self.momentum

        if self.training:
            training = True
        else:
            training = (self.running_mean is None) and (self.running_var is None)

        return momentum, training

    def _get_running_stats(self):
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        return running_mean, running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        x = functional.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training,
            momentum,
            self.eps,
        )
        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "silu":
            return functional.silu(x, inplace=True)
        elif self.activation == "h_swish":
            return functional.hardswish(x, inplace=True)
        elif self.activation == "gelu":
            return functional.gelu(x)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(
                x, negative_slope=self.activation_param, inplace=True
            )
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ABN, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)

        
class ActivatedAffine(ABN):
    """Drop-in replacement for ABN which performs inference-mode BN + activation"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super(ActivatedAffine, self).__init__(num_features, eps, momentum, affine, activation, activation_param)

    @staticmethod
    def _broadcast_shape(x):
        out_size = []
        for i, s in enumerate(x.size()):
            if i != 1:
                out_size.append(1)
            else:
                out_size.append(s)
        return out_size

    def forward(self, x):
        
        inv_var = torch.rsqrt(self.running_var + self.eps)
        
        if self.affine:
            alpha = self.weight * inv_var
            beta = self.bias - self.running_mean * alpha
        else:
            alpha = inv_var
            beta = - self.running_mean * alpha

        x.mul_(alpha.view(self._broadcast_shape(x)))
        x.add_(beta.view(self._broadcast_shape(x)))

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class ActivatedGroupNorm(ABN):
    """GroupNorm + activation function compatible with the ABN interface"""

    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True, activation="leaky_relu", activation_param=0.01):
        super(ActivatedGroupNorm, self).__init__(num_channels, eps, affine=affine, activation=activation,
                                                 activation_param=activation_param)
        self.num_groups = num_groups

        # Delete running mean and var since they are not used here
        delattr(self, "running_mean")
        delattr(self, "running_var")

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))



def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str


def scheduler_from_config(scheduler_config, optimizer, epoch_length):
    assert scheduler_config["type"] in ("linear", "step", "exp", "poly", "multistep")

    params = scheduler_config.getstruct("params")

    if scheduler_config["type"] == "linear":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")

        beta = float(params["from"])
        alpha = float(params["to"] - beta) / count

        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: it * alpha + beta)

    elif scheduler_config["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        params["step_size"],
                                        params["gamma"])
    
    elif scheduler_config["type"] == "exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                                gamma=params["gamma"])
    elif scheduler_config["type"] == "poly":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: (1 - float(it) / count) ** params["gamma"])

    elif scheduler_config["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             params["milestones"],
                                             params["gamma"])

    else:
        raise ValueError("Unrecognized scheduler type {}, valid options: 'linear', 'step', 'poly', 'multistep'"
                         .format(scheduler_config["type"]))

    if scheduler_config.getint("burn_in_steps") != 0:
        scheduler = lr_scheduler.BurnInLR(scheduler,
                                          scheduler_config.getint("burn_in_steps"),
                                          scheduler_config.getfloat("burn_in_start"))

    return scheduler


def norm_act_from_config(body_config):
    """Make normalization + activation function from configuration

    Available normalization modes are:
      - `bn`: Standard In-Place Batch Normalization
      - `syncbn`: Synchronized In-Place Batch Normalization
      - `syncbn+bn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Standard In-Place
        Batch Normalization in the "dynamic" parts
      - `gn`: Group Normalization
      - `syncbn+gn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Group Normalization
        in the "dynamic" parts
      - `off`: No normalization (preserve scale and bias parameters)

    The "static" part of the network includes the backbone, FPN and semantic segmentation components, while the
    "dynamic" part of the network includes the RPN, detection and instance segmentation components. Note that this
    distinction is due to historical reasons and for back-compatibility with the CVPR2019 pre-trained models.

    Parameters
    ----------
    body_config
        Configuration object containing the following fields: `normalization_mode`, `activation`, `activation_slope`
        and `gn_groups`

    Returns
    -------
    norm_act_static : callable
        Function that returns norm_act modules for the static parts of the network
    norm_act_dynamic : callable
        Function that returns norm_act modules for the dynamic parts of the network
    """
    mode = body_config["normalization_mode"]
    activation = body_config["activation"]
    slope = body_config.getfloat("activation_slope")
    groups = body_config.getint("gn_groups")
    
    if mode == "ln":
        norm_act_static = norm_act_dynamic = partial(ALN, activation=activation, activation_param=slope)
        
    elif mode == "bn":
        norm_act_static = norm_act_dynamic = partial(ABN, activation=activation, activation_param=slope)

    elif mode == "gn":
        norm_act_static = norm_act_dynamic = partial(ActivatedGroupNorm, num_groups=groups, activation=activation, activation_param=slope)

    elif mode == "off":
        norm_act_static = norm_act_dynamic = partial(ActivatedAffine, activation=activation, activation_param=slope)

    elif mode == "br+bg":
        norm_act_static     = partial(ABN, activation="relu", activation_param=slope)
        norm_act_dynamic    = partial(ABN, activation="gelu", activation_param=slope)

    else:
        raise ValueError("Unrecognized normalization_mode {}, valid options: 'bn', 'syncbn', 'syncbn+bn', 'gn', "
                         "'syncbn+gn', 'off'".format(mode))

    return norm_act_static, norm_act_dynamic


def freeze_params(module):
    """Freeze all parameters of the given module"""
    for p in module.parameters():
        p.requires_grad_(False)


def all_reduce_losses(losses):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in losses.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.cat([v.view(1) for v in values], dim=0)
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.view(())) for k, v in zip(names, values))


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list


from torch.nn import *

LINEAR_CONV_LAYERS = [
    Identity, Linear, Bilinear, LazyLinear,
    Conv1d, Conv2d, Conv3d, \
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, \
    LazyConv1d, LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d
    ]

ACTIVATIONS_LAYERS = [
    Threshold, ReLU, Hardtanh, ReLU6, Sigmoid, Tanh,
    Softmax, Softmax2d, LogSoftmax, ELU, SELU, CELU, GELU, Hardshrink, LeakyReLU, LogSigmoid, \
    Softplus, Softshrink, MultiheadAttention, PReLU, Softsign, Softmin, Tanhshrink, RReLU, GLU, \
    Hardsigmoid, Hardswish, SiLU, Mish
    ]

POOLING_LAYERS = [
    AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d, \
    MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, FractionalMaxPool2d, FractionalMaxPool3d, LPPool1d, LPPool2d, \
    AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d]

NORM_LAYERS = [
    BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, \
    LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, \
    LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d,
    LocalResponseNorm, CrossMapLRN2d, LayerNorm, GroupNorm]


OTHER_LAYERS = LINEAR_CONV_LAYERS + ACTIVATIONS_LAYERS + POOLING_LAYERS

