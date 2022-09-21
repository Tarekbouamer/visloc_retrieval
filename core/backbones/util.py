import torch.nn as nn

from core.utils.misc import ABN

CONV_PARAMS = ["weight"]
BN_PARAMS = ["weight", "bias", "running_mean", "running_var"]
FC_PARAMS = ["weight", "bias"]

def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


#TODO: is init implmeneted elsewhere so can be delleted from here

def init_weights(model, config):

    for name, m in model.named_modules():
        
        if isinstance(m, nn.Conv2d):
            init_fn = getattr(nn.init, config.get("initializer") + '_')
            
            # Xavier or Orthogonal
            if config.get("initializer").startswith("xavier") or config.get("initializer") == "orthogonal":

                gain = config.getfloat("weight_gain_multiplier")
                if config.get("activation") == "relu" or config.get("activation") == "elu":
                    gain *= nn.init.calculate_gain("relu")
                elif config.get("activation") == "leaky_relu":
                    gain *= nn.init.calculate_gain("leaky_relu", config.getfloat("activation_slope"))
                init_fn(m.weight, gain)

            # Kaiming He
            elif config.get("initializer").startswith("kaiming"):
                if config.get("activation") == "relu" or config.get("activation") == "elu":
                    init_fn(m.weight, 0)
                else:
                    init_fn(m.weight, config.getfloat("activation_slope"))
            # Bias
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
            
            
