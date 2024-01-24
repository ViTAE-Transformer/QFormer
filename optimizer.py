# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if hasattr(model, 'learning_rate_10xless_keywords'):
        learning_rate_10xless_keywords = model.learning_rate_10xless_keywords()
    else:
        learning_rate_10xless_keywords = []
    parameters = set_lr_and_weight_decay(model, config, skip, skip_keywords, learning_rate_10xless_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer

def set_lr_and_weight_decay(model, config, skip_list=(), skip_keywords=(), learning_rate_10xless_keywords=()):
    has_decay = []
    has_decay_10xless = []
    no_decay = []
    no_decay_10xless = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            if check_keywords_in_name(name, learning_rate_10xless_keywords):
                no_decay_10xless.append(param)
            else:
                no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            if check_keywords_in_name(name, learning_rate_10xless_keywords):
                has_decay_10xless.append(param)
            else:
                has_decay.append(param)
    return [{'params': has_decay},
            {'params': has_decay_10xless, 'lr': 0.1*config.TRAIN.BASE_LR},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': no_decay_10xless, 'weight_decay': 0., 'lr': 0.1*config.TRAIN.BASE_LR}
            ]

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
