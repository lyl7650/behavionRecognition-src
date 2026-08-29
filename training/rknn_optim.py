"""Optimizer grouping shared by the opt-in RKNN training copies."""

import torch.nn as nn


def adamw_groups_with_bn_gamma_decay(model, *, base_weight_decay, bn_gamma_weight_decay):
    """Keep BN bias unregularized but mildly decay its scale (gamma)."""
    regular, bn_gamma, no_decay = [], [], []
    bn_weight_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.weight is not None}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in bn_weight_ids:
            bn_gamma.append(param)
        elif param.ndim <= 1:
            no_decay.append(param)
        else:
            regular.append(param)
    return [
        {"params": regular, "weight_decay": base_weight_decay},
        {"params": bn_gamma, "weight_decay": bn_gamma_weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
