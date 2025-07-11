import os
import math
import time
import random
import torch
import argparse
import builtins
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from typing import Tuple
import backbone as archs
from thop import profile
from utils.metrics import accuracy
from utils.meter import AverageMeter, ProgressMeter, InstantMeter
from utils.dist_init import random_choice, set_seed
from utils.config import cfg, update_config_from_file
from utils.samplers import RASampler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils.model import unwrap_model
from data.dataset import init_imagenet, get_source_dataset, get_target_dataset
from torch.distributed import init_process_group, destroy_process_group
from data.augmentation import val_aug
import torch._dynamo
torch._dynamo.config.suppress_errors = True
parser = argparse.ArgumentParser()
# Model Settings
parser.add_argument("--port", type=int, default=23457)
parser.add_argument("--arch", type=str, default="ResNet")
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--eval_classes", type=int, default=1000)

# config file
parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str,default="experiments/supernet/supernet-B.yaml")

# Training Settings
parser.add_argument("--seed", type=int, default=0, help="training seed")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--alpha", type=float, default=0.0)
# parser.add_argument("--smoothing", type=float, default=0.0)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--train_full", default=False, action="store_true")
parser.add_argument("--eval_only", default=False, action="store_true")
parser.add_argument("--freeze_encoder", default=False, action="store_true")
parser.add_argument("--w_rcc", default=False, action="store_true")
parser.add_argument("--finetune_ckpt", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--pretrained_weights",
    type=str,
    default=None,
    # choices=[None, "IMAGENET1K_V1", "IMAGENET1K_V2"],
)
parser.add_argument('--mode', type=str, default='retrain', choices=['super', 'retrain'], help='mode of AutoFormer')
# Supernet Setting
parser.add_argument("--layers", type=int, default=20, help="number of choice_block layers in CNNs or operations in ViT")
parser.add_argument("--num_choices", type=int, default=4, help="number of choices per layer in CNNs or per op in ViT")
parser.add_argument("--use_lmdb", action='store_true', help="whether or not use LMDB dataset")

# Dataset parameters
parser.add_argument('--data-base-path', default='~/work/DomainBed/domainbed/data', type=str,
                    help='dataset path')
parser.add_argument('--task', default='OfficeHome', choices=['OfficeHome', 'PACS', 'VLCS','DomainNet'],
                    type=str, help='multi-domain dataset path')
parser.add_argument("--source", type=str, nargs='+', default=['Art'])
parser.add_argument("--target", type=str, nargs='+', default=['Clipart', 'Product', 'RealWorld'])
parser.add_argument("--train_data_ratio", type=float, default=0.7)

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

parser.add_argument('--repeated-aug', action='store_true')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

args = parser.parse_args()
set_seed(args.seed)
depth_list = [12,13,14]
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

@torch.no_grad()
def batch_process_label(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    class0 = label == 0
    class1 = label == 1
    stack_label = torch.stack([class0, class1], dim=1)   # label for ce_loss, stack_label for dice loss
    stack_label = (stack_label > 0.5).float()
    return label.long(), stack_label


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, i: int, iteration_per_epoch: int
) -> None:
    warm_up = max(1, args.epochs // 20)
    base_lr = args.lr
    epochs = args.epochs
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr * T / warmup_iters
    else:
        min_lr = base_lr / 1000
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = (
            0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr)
            + min_lr
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    mixup_fn: Mixup,
    mode: str,
    choices=None,
    retrain_config=None,
) -> Tuple[float, float]:
    batch_time = AverageMeter("Time", ":6.3f")
    curr_lr = InstantMeter("LR", ":6.5f")
    ce_losses = AverageMeter("CE", ":.4e")
    train_acc = AverageMeter("ACC", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, curr_lr, ce_losses, train_acc],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    if args.freeze_encoder:
        if hasattr(model, "module"):
            model.module.encoder.eval()
        else:
            model.encoder.eval()

    iteration_per_epoch = len(train_loader)
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).long()
        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        # print(ce_target.shape)
        loss = 0
        with torch.cuda.amp.autocast(enabled=True):
            output = model(input)
            ce_loss = criterion(output, target)
            loss += ce_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        curr_lr.update(optimizer.param_groups[0]["lr"])
        ce_losses.update(loss.item(), input.size(0))
        
        acc = 0
        if args.num_classes > 2:
            prec1, _ = accuracy(output, target, topk=(1,5))
        else:
            prec1, = accuracy(output, target, topk=(1,))
        acc += prec1
        
        train_acc.update(acc.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)

    return ce_losses.avg, train_acc.avg


@torch.no_grad()
def validate(
    val_loader: torch.utils.data.DataLoader, model: nn.Module, choices=None, mode='super', retrain_config=None
) -> Tuple[float, float]:

    val_losses = AverageMeter("CE", ":.4e")
    val_acc = AverageMeter("Acc",":.4e")
    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).long()
        output = model(input)
        
        loss = criterion(output, target)

        if args.num_classes > 2:
            prec1, _ = accuracy(output, target, topk=(1,5))
        else:
            prec1, = accuracy(output, target, topk=(1,))

        val_losses.update(loss.item(), input.size(0))
        val_acc.update(prec1.item(), input.size(0))

    return val_losses.avg, val_acc.avg


def main():
    init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    cudnn.benchmark = True

    world_size = dist.get_world_size()
    bs = args.bs // world_size  # 256 // 2 = 128
    args.lr *= args.bs / args.bs

    if args.mode == 'super':
        train_dataset, val_dataset = init_imagenet(
            data_dir = '/home/qinghua_mao/work/ImageNet_1k/train',
            pretrained = True,
            args=args
        )
    else:
        train_dataset, val_dataset, _ = get_target_dataset(args)
    if args.repeated_aug:
        train_sampler = RASampler(train_dataset)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        # shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True,
    )
      
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=bs,
        # shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        persistent_workers=True,
    )

    # create model
    print("=> creating model %s" % args.arch)
    update_config_from_file(args.cfg)
    # must include necessary config arguments
    model = archs.__dict__[args.arch](
        img_size=args.img_size,
        num_classes=args.num_classes,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,

    )
    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}

    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    model = torch.compile(model)

    if args.mixup > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # mixup 
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    
    if not os.path.exists("checkpoints") and rank == 0:
        os.makedirs("checkpoints")

    if args.finetune_ckpt is not None:
        print("loading checkpoint from {}".format(args.finetune_ckpt))
        finetune_ckpt = "Pretrained_Models/{}.pth".format(args.finetune_ckpt)
        finetune_ckpt = torch.load(finetune_ckpt, map_location="cpu")
        # model.load_state_dict(finetune_ckpt["model"], strict=False)
        model_dict = model.state_dict()
        pretrained_dict = finetune_ckpt["model"]
        # new_state_dict={k:v if v.size()==current_ckpt[k].size()  else  current_ckpt[k] for k,v in zip(current_ckpt.keys(), loaded_ckpt.values())}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        # optimizer.load_state_dict(finetune_ckpt["optimizer"])
        # scaler.load_state_dict(finetune_ckpt["scaler"])

    checkpoint_path = "checkpoints/{}.pth.tar".format(args.checkpoint)
    print("checkpoint_path:", checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
        best_loss = 1e4
        best_acc = 0

    retrain_config = None
    if args.mode == 'retrain' and "RETRAIN" in cfg:
        retrain_config = {'layer_num': cfg.RETRAIN.DEPTH, 'embed_dim': [cfg.RETRAIN.EMBED_DIM]*cfg.RETRAIN.DEPTH,
                          'num_heads': cfg.RETRAIN.NUM_HEADS,'mlp_ratio': cfg.RETRAIN.MLP_RATIO}
    if args.eval_only:
        val_loss, val_acc = validate(val_loader=val_loader, model=model_without_ddp, mode='retrain', retrain_config=retrain_config)
        print(
            "Epoch: {} val_loss {:.4f} - val_acc {:.4f}".format(
                start_epoch - 1, val_loss, val_acc
            )
        )
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            _, _ = train_one_epoch(train_loader=train_loader, model=model, criterion=criterion, 
                            optimizer=optimizer, scaler=scaler, epoch=epoch, mixup_fn=mixup_fn, choices=choices, mode='retrain', retrain_config=retrain_config)
            if val_loader is not None:
                val_loss, val_acc = validate(val_loader=val_loader, model=model_without_ddp, choices=choices, mode='retrain', retrain_config=retrain_config)
                best_loss = min(best_loss, val_loss)
                best_acc = max(best_acc, val_acc)

                print(
                    "Epoch: {} val_loss {:.4f} - val_acc {:.4f} - best_loss {:.4f} - best_acc {:.4f}".format(
                        epoch, val_loss, val_acc, best_loss, best_acc
                    )
                )

            if rank == 0:
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_loss": float(best_loss),
                    "best_acc": float(best_acc),
                    "epoch": epoch + 1,
                }
                torch.save(state_dict, checkpoint_path)
    destroy_process_group()


if __name__ == "__main__":
    main()
