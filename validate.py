# --------------------------------------------------------
# Swin Transformer
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

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, GET_WORLD_SIZE, GET_RANK, is_main_process, get_wandb_keys

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
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

    parser.add_argument('--instance_tokens', default=None, type=int, nargs='+', 
        help='extra instance tokens at each level')
    parser.add_argument('--EM_iters', type=int, default=None)
    parser.add_argument('--EM_factor', type=float, default=None)
    parser.add_argument('--enable_wandb', action='store_true',
                        help='whether to enable wandb to monitor training')

    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--real", action='store_true')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, real=args.real)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.EMA.ENABLE_EMA:
        model_ema = ModelEma(
            model,
            decay=config.EMA.EMA_DECAY,
            device='cpu' if config.EMA.EMA_FORCE_CPU else '',
            resume='')
        if hasattr(model_ema.ema, 'module'):
            model_ema_without_ddp = model_ema.ema.module
        else:
            model_ema_without_ddp = model_ema.ema
        logger.info('enable EMA model')
    else:
        model_ema = None
        model_ema_without_ddp = None
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # if dist.is_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    #     model_without_ddp = model.module
    # else:
        # model = torch.nn.DataParallel(model, )
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.ENABLE_WANDB and is_main_process():
        WANDB_API_KEY = get_wandb_keys('netrc')
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.init(project='deformable_WDA', entity="qimingzhang", settings=wandb.Settings(_disable_stats=True))
        wandb.config = config
        wandb.run.name = '-'.join([args.cfg.split('/')[-1].split('.')[0], f'tag_{args.tag}'])

    # log with wandb
    # if utils.get_rank() == 0:
    #     if args.wandb:
    #         wandb.init(config=args, project="pnp-detr")
    #         wandb.run.name = '_'.join([
    #             args.dataset_file, os.path.basename(args.output_dir), 'bs{}x{}'.format(args.world_size, args.batch_size),
    #             'seed{}'.format(args.seed),
    #         ])
    #     else:
    #         warnings.warn("wandb is turned off")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_epoch = 0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, model_ema_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, dataset_val, args.real)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1, acc5, loss = validate(config, data_loader_val, model_ema.ema, dataset_val, args.real)
            logger.info(f"Accuracy of the ema network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
        # if config.EVAL_MODE:
        return


def train_one_epoch(config, model, model_ema, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    pass


@torch.no_grad()
def validate(config, data_loader, model, dataset=None, real=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    if real:
        real_acc1_meter = AverageMeter()
        real_acc5_meter = AverageMeter()
        end = time.time()
        for idx, (images, target, real_target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if dist.is_initialized():
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)

            dataset.add_result(output, real_target)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            real_acc1_meter.update(dataset.get_accuracy(1), target.size(0))
            real_acc5_meter.update(dataset.get_accuracy(5), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Real Acc@1 {real_acc1_meter.val:.3f} ({real_acc1_meter.avg:.3f})\t'
                    f'Real Acc@5 {real_acc5_meter.val:.3f} ({real_acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        logger.info(f' * Real Acc@1 {real_acc1_meter.avg:.3f} Real Acc@5 {real_acc5_meter.avg:.3f}')
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if dist.is_initialized():
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    args, config = parse_option()

    print("config.AMP_OPT_LEVEL:", config.AMP_OPT_LEVEL)
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ['WORLD_SIZE'])
        # print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    rank = -1
    world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # if args.distributed:
        # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        # torch.distributed.barrier()

    seed = config.SEED + GET_RANK()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    model_ema_decay = config.EMA.EMA_DECAY
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

    main(config, args)
