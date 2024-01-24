# --------------------------------------------------------
# Vision Transformer with Quadrangle Attention
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath('.'))
import time
import argparse
import datetime
import numpy as np
from PIL import Image
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from einops import rearrange
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms.functional as F

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, GET_WORLD_SIZE, GET_RANK, is_main_process, get_wandb_keys, load_pretrained

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

COCO_MEAN=(123.675, 116.28, 103.53)
COCO_STD=(58.395, 57.12, 57.375)

def parse_option():
    parser = argparse.ArgumentParser('Vision Transformer with Quadrangle Attention training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--disable_resume_optimizer', action='store_true', help='load only the pretrained weights')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--distributed', action='store_true', )
    parser.add_argument('--epochs', type=int, help='epochs for training')
    parser.add_argument('--warmup_epochs', type=int, help='warmup epochs')
    parser.add_argument('--enable_wandb', action='store_true',
                        help='whether to enable wandb to monitor training')
    parser.add_argument('--coords_lambda', type=float, default=None,
                        help='coords_lambda for pixel location regularization')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='drop path rate')

    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')

    # visual input
    parser.add_argument('--input_imgs', type=str, default=None)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def generate_colors(n_colors, seed=47):
    """
    随机生成颜色
    """
    np.random.seed(seed)
    color_list = []
    for i in range(n_colors):
        color = (np.random.random((1, 3)) * 0.8).tolist()[0]
        color = [int(j * 255) for j in color]
        color_list.append(color)

    return color_list


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    # if config.EMA.ENABLE_EMA:
    #     model_ema = ModelEma(
    #         model,
    #         decay=config.EMA.EMA_DECAY,
    #         device='cpu' if config.EMA.EMA_FORCE_CPU else '',
    #         resume='')
    #     if hasattr(model_ema.ema, 'module'):
    #         model_ema_without_ddp = model_ema.ema.module
    #     else:
    #         model_ema_without_ddp = model_ema.ema
    #     logger.info('enable EMA model')
    # else:
    # model_ema = None
    # model_ema_without_ddp = None
    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        # model = torch.nn.DataParallel(model, )
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    # max_accuracy = 0.0
    # max_epoch = 0

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    attention_distance = []

    def forward_hook(module, data_in, data_out):
        attention_distance.append(data_in)

    for layer in model.layers:
        for block in layer.blocks:
            block.attn.identity_distance.register_forward_hook(forward_hook)

    model.eval()
    img_path = config.VISUAL.input_imgs
    if img_path is not None and os.path.isdir(img_path):
        img_paths = []
        for root_path, _, files in os.walk(img_path):
            for _file in files:
                img_paths.append(os.path.join(root_path, _file))

    if True:
        # for layer in model.layers:
        #     for block in layer.blocks:
        #         offsets.append([])
        all_attention_distance = [torch.tensor([]) for _ in range(sum(config.MODEL.SWIN.DEPTHS))]
        # scales = [torch.tensor([]).cuda() for _ in range(sum(config.MODEL.SWIN.DEPTHS))]
        # shear = [torch.tensor([]).cuda() for _ in range(sum(config.MODEL.SWIN.DEPTHS))]
        # projc = [torch.tensor([]).cuda() for _ in range(sum(config.MODEL.SWIN.DEPTHS))]
        # rotation = [torch.tensor([]).cuda() for _ in range(sum(config.MODEL.SWIN.DEPTHS))]
        for _, (samples, targets) in enumerate(data_loader_val):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            attention_distance = []
            with torch.no_grad():
                output = model(samples)
            for idx in range(len(attention_distance)):
                distance = attention_distance[idx][0]
                all_attention_distance[idx] = torch.cat((all_attention_distance[idx], distance.reshape(-1).cpu()), dim=-1)
                # scales[idx] = torch.cat((scales[idx], sampling_scales.reshape(-1)), dim=-1)
                # shear[idx] = torch.cat((shear[idx], sampling_shear.reshape(-1)), dim=-1)
                # projc[idx] = torch.cat((projc[idx], sampling_projc.reshape(-1)), dim=-1)
                # rotation[idx] = torch.cat((rotation[idx], sampling_rotation.reshape(-1)), dim=-1)
            # if _ > 200:
            #     break
            # if _ % 1000 == 0:
            #     print(f"{_*B} samples")
        # np.save("offset", np.array(offsets))
        # np.save("scales", np.array(scales))
        # np.save("shear", np.array(shear))
        # np.save("projc", np.array(projc))
        # np.save("rotation", np.array(rotation))
        torch.save(all_attention_distance, 'all_attention_distance.pth')
        print('finished')


if __name__ == '__main__':
    args, config = parse_option()

    print("config.AMP_OPT_LEVEL:", config.AMP_OPT_LEVEL)
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

    seed = config.SEED + GET_RANK()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * GET_WORLD_SIZE() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * GET_WORLD_SIZE() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * GET_WORLD_SIZE() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    model_ema_decay = config.EMA.EMA_DECAY ** (config.DATA.BATCH_SIZE * GET_WORLD_SIZE() / 512.0)
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.EMA.EMA_DECAY = model_ema_decay
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=GET_RANK(), name=f"{config.MODEL.NAME}")

    if GET_RANK() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
