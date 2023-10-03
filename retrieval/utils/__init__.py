
from torch.nn import *  # noqa: F403

LINEAR_CONV_LAYERS = [Identity, Linear, Bilinear, LazyLinear, Conv1d, Conv2d, Conv3d,\
                      ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, LazyConv1d,\
                      LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d,\
                      LazyConvTranspose3d ]  # noqa: F405

ACTIVATIONS_LAYERS = [Threshold, ReLU, Hardtanh, ReLU6, Sigmoid, Tanh,Softmax,\
                      Softmax2d,LogSoftmax, ELU, SELU, CELU, GELU, Hardshrink,\
                      LeakyReLU, LogSigmoid, Softplus, Softshrink, MultiheadAttention,\
                      PReLU, Softsign, Softmin,Tanhshrink, RReLU, GLU,Hardsigmoid, \
                      Hardswish, SiLU, Mish ] # noqa: F405

POOLING_LAYERS = [AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d,\
                  MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, FractionalMaxPool2d,\
                  FractionalMaxPool3d, LPPool1d, LPPool2d,AdaptiveMaxPool1d, \
                  AdaptiveMaxPool2d, AdaptiveMaxPool3d, AdaptiveAvgPool1d,\
                  AdaptiveAvgPool2d, AdaptiveAvgPool3d] # noqa: F405

NORM_LAYERS = [BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, LazyBatchNorm1d, \
               LazyBatchNorm2d, LazyBatchNorm3d, InstanceNorm1d, InstanceNorm2d, \
               InstanceNorm3d, LazyInstanceNorm1d, LazyInstanceNorm2d,\
               LazyInstanceNorm3d, LocalResponseNorm, CrossMapLRN2d, LayerNorm, \
               GroupNorm ] # noqa: F405


OTHER_LAYERS = LINEAR_CONV_LAYERS + ACTIVATIONS_LAYERS + POOLING_LAYERS

__all__ = ['LINEAR_CONV_LAYERS', 'ACTIVATIONS_LAYERS',
           'POOLING_LAYERS', 'NORM_LAYERS', 'OTHER_LAYERS']
