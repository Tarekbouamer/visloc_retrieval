
import os

import gdown
import timm
import torch.nn as nn
from core.registry.factory import load_state_dict
from core.registry.register import get_pretrained_cfg
from omegaconf import OmegaConf

from retrieval.models.modules.pools import GeM

from .misc import _cfg, register_retrieval

# @torch.no_grad()
# def _init_model(args, cfg, model, sample_dl):

#     # eval
#     if model.training:
#         model.eval()

#     # options
#     device = get_device()

#     #
#     model_head = model.head
#     inp_dim = model_head.inp_dim
#     out_dim = model_head.out_dim

#     # prepare query loader
#     logger.info(
#         f'extracting descriptors for PCA {inp_dim}--{out_dim} for {len(sample_dl)}')

#     # extract vectors
#     vecs = []
#     for _, batch in tqdm(enumerate(sample_dl), total=len(sample_dl)):

#         # upload batch
#         batch = {k: batch[k].cuda(device=device, non_blocking=True)
#                  for k in INPUTS}
#         pred = model(**batch, do_whitening=False)

#         vecs.append(pred['features'].cpu().numpy())

#         del pred

#     # stack
#     vecs = np.vstack(vecs)

#     logger.info('compute PCA')

#     m, P = PCA(vecs)
#     m, P = m.T, P.T

#     # create layer
#     layer = deepcopy(model_head.whiten)
#     data_size = layer.weight.data.size()

#     #
#     num_d = layer.weight.shape[0]

#     #
#     projection = torch.Tensor(P[: num_d, :]).view(data_size)
#     projected_shift = - \
#         torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()

#     #
#     layer.weight.data = projection
#     layer.bias.data = projected_shift[:num_d]

#     # save layer to whithen_path
#     layer_path = os.path.join(args.directory, "whiten.pth")
#     torch.save(layer.state_dict(), layer_path)

#     logger.info(f"save whiten layer: {layer_path}")

#     # load
#     model.head.whiten.load_state_dict(layer.state_dict())

#     logger.info("pca done")

#     return


# def parameter_groups(self, cfg):
#     """
#         Return torch parameter groups
#     """

#     # base
#     LR = cfg.optimizer.lr
#     WEIGHT_DECAY = cfg.optimizer.weight_decay

#     # base layer
#     layers = [self.body, self.head.whiten]

#     # base params
#     params = [{
#         'params':          [p for p in x.parameters() if p.requires_grad],
#         'lr':              LR,
#         'weight_decay':    WEIGHT_DECAY} for x in layers]

#     # 10x faster, no regularization
#     if self.head.pool:
#         params.append({
#             'params':       [p for p in self.head.pool.parameters() if p.requires_grad],
#             'lr':           10*LR,
#             'weight_decay': 0.}
#         )

#     return params


# def __forward__(self, img, scales=[1], do_whitening=True):

#     #
#     features = []

#     # -->
#     for scale in scales:

#         # resize
#         img_s = self.__resize__(img, scale=scale)

#         # assert size within boundaries
#         if self.__check_size__(img_s):
#             continue

#         # -->
#         preds = self.forward(img_s, do_whitening)

#         #
#         features.append(preds['features'])

#     # sum over scales
#     return self.__sum_scales(features)


# def __sum_scales(self, features):
#     """ sum over scales """
#     #
#     desc = torch.zeros(
#         (1, features[0].shape[1]), dtype=torch.float32, device=features[0].device)

#     #
#     for vec in features:
#         desc += vec

#     # normalize
#     desc = functional.normalize(desc, dim=-1)

#     preds = {
#         'features': desc
#     }
#     return preds


# def ___create_model(variant, body_name, head_name, cfg=None, pretrained=True, feature_scales=[1, 2, 3, 4], **kwargs):

#     # assert
#     assert body_name in timm.list_models(
#         pretrained=True), f"model: {body_name}  not implemented timm models yet!"

#     # default cfg
#     default_cfg = get_pretrained_cfg(variant)
#     out_dim = default_cfg.pop('out_dim', None)
#     frozen = default_cfg.pop("frozen", [])

#     body_channels = body.feature_info.channels()
#     body_reductions = body.feature_info.reduction()
#     body_module_names = body.feature_info.module_name()

#     # output dim
#     body_dim = body_channels[-1]

#    # freeze layers
#     if len(frozen) > 0:
#         frozen_layers = [body_module_names[item] for item in frozen]
#         logger.info(f"frozen layers {frozen_layers}")
#         freeze(body, frozen_layers)

#     # reduction
#     assert out_dim <= body_dim, (
#         f"reduction {out_dim} has to be less than or equal to input dim {body_dim}")

#     # head
#     head = create_head(head_name,
#                        inp_dim=body_dim,
#                        out_dim=out_dim,
#                        **kwargs)

#     # model
#     model = GemNet(body, head, init_model=_init_model)

#     #
#     if pretrained:
#         save_path = "hub" + "/" + variant + ".pth"
#         state_dict = load_state_dict(save_path)["state_dict"]
#         model.body.load_state_dict(state_dict["body"])
#         model.head.load_state_dict(state_dict["head"])

#     logger.info(
#         f"body channels:{body_channels}  reductions:{body_reductions}   layer_names: {body_module_names}")

#     #
#     if cfg:
#         OmegaConf.update(cfg, 'global.global_dim', str(out_dim))

#     return model

class GemHead(nn.Module):
    def __init__(self, inp_dim, out_dim, p=3.0):
        super(GemHead, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # pooling
        self.pool = GeM(p=p)

        # whitening
        self.whiten = nn.Linear(inp_dim, out_dim, bias=True)

    def forward(self, x, do_whitening=True):

        # pooling
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        # whithen
        if do_whitening:
            x = self.whiten(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)

        return {'features': x}


class GemNet(nn.Module):

    def __init__(self, cfg):
        super(GemNet, self).__init__()

        # cfg
        self.cfg = OmegaConf.create(cfg)


        # create backbone FIXME: add pretrained correctly
        self.body = timm.create_model(self.cfg.backbone,
                                      features_only=True,
                                      out_indices=self.cfg.feature_scales,
                                      #   pretrained=self.cfg.pretrained
                                      )
        # inp dim
        inp_dim = self.body.feature_info.channels()[-1]

        # create head
        self.head = GemHead(inp_dim=inp_dim,
                            out_dim=self.cfg.out_dim,
                            p=self.cfg.p)

    def forward(self, img, do_whitening=True):
        # body
        x = self.body(img)
        x = x[-1]

        # head
        return self.head(x, do_whitening)

    def extract(self, img):
        return self.forward(img, do_whitening=True)


default_cfgs = {
    'sfm_resnet50_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1gFRNJPILkInkuCZiCHqjQH_Xa2CUiAb5',
             backbone="resnet50", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
    'sfm_resnet50_c4_gem_1024':
        _cfg(drive='https://drive.google.com/uc?id=1ber3PbTF4ZWAmnBuJu5AEp2myVJFNM7F',
             backbone="resnet50", feature_scales=[1, 2, 3], out_dim=1024, p=3.0),
    'sfm_resnet101_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=10CqmzZE_XwRCyoiYlZh03tfYk1jzeziz',
             backbone="resnet101", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
    'sfm_resnet101_c4_gem_1024':
        _cfg(drive='https://drive.google.com/uc?id=1uYYuLqqE9TNgtmQtY7Mg2YEIF9VkqAYz',
             backbone="resnet101", feature_scales=[1, 2, 3], out_dim=1024, p=3.0),
    'gl18_resnet50_gem_2048':
        _cfg(drive='https://drive.google.com/uc?id=1AaS4aXe2FYyi-iiLetF4JAo0iRqKHQ2Z',
             backbone="resnet50", feature_scales=[1, 2, 3, 4], out_dim=2048, p=3.0),
}


def load_from_drive(variant, pretrained_drive):
    # name hub folder
    save_folder = "hub"
    save_path = save_folder + "/" + variant + ".pth"

    # create fodler
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # download from gdrive if weights not found
    if not os.path.exists(save_path):
        save_path = gdown.download(
            pretrained_drive, save_path, quiet=False, use_cookies=False)

    #  load from drive
    state_dict = load_state_dict(save_path)

    return state_dict


def _create_model(variant, cfg: dict = {}, pretrained: bool = True, **kwargs: dict) -> nn.Module:
    """Creates a model from a given configuration.
    """

    # default cfg
    model_cfg = get_pretrained_cfg(variant)
    cfg = {**model_cfg, **cfg}

    # create model
    model = GemNet(cfg=cfg)

    # load pretrained weights
    if pretrained:
        state_dict = load_from_drive(variant, cfg["drive"])["state_dict"]
        model.body.load_state_dict(state_dict["body"])
        model.head.load_state_dict(state_dict["head"])

    return model


@register_retrieval
def sfm_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_gem_2048', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet50_c4_gem_1024(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-50 with GeM model, only 4 features scales
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet50_c4_gem_1024', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet101_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_gem_2048', cfg, pretrained, **model_args)


@register_retrieval
def sfm_resnet101_c4_gem_1024(cfg=None, pretrained=True, **kwargs):
    """Constructs a SfM-120k ResNet-101 with GeM model, only 4 features scales
    """
    model_args = dict(**kwargs)
    return _create_model('sfm_resnet101_c4_gem_1024', cfg, pretrained, **model_args)


@register_retrieval
def gl18_resnet50_gem_2048(cfg=None, pretrained=True, **kwargs):
    """Constructs a gl18 ResNet-50 with GeM model.
    """
    model_args = dict(**kwargs)
    return _create_model('gl18_resnet50_gem_2048', cfg, pretrained, **model_args)
