import os
import json
from openai import OpenAI
import math
import time
import random
from thop import profile
import numpy as np
from decimal import Decimal
import argparse
import torch.distributed as dist
import torch.optim as optim
import builtins
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from typing import Tuple, List
import backbone as archs
from utils.metrics import accuracy
from utils.meter import AverageMeter, ProgressMeter, InstantMeter
from utils.dist_init import set_seed, get_rank, get_world_size
from utils.config import cfg, update_config_from_file
from utils.samplers import RASampler
from learn2learn.algorithms import MAML
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils.model import unwrap_model
from data.dataset import get_source_dataset, get_target_dataset, get_augmented_dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.augmentation import TranAugment, val_aug
import torch._dynamo
torch._dynamo.config.suppress_errors = True

llm_params = {
    "qwen": {
        "name": "qwen-plus",
        "api_key": "xxx",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "deepseek": {
        "name": "deepseek-chat", 
        "api_key": "sk-ddfbfa23dc7248ecb00bf175eb1648c0",
        "base_url": "https://api.deepseek.com"},
    "deepseek-v3": {
        "name": "DeepSeek-V3", 
        "api_key": "xxx",
        "base_url": "https://api.sensenova.cn/compatible-mode/v1/"},
}
parser = argparse.ArgumentParser()

# openai settings
parser.add_argument('--openai_key', type=str, default="xxx")
# parser.add_argument('--openai_organization', type=str, required=True)

# Supernet Setting
parser.add_argument("--arch", type=str, default="vit_small_patch16_224", help="supernet architecture")
parser.add_argument("--layers", type=int, default=20, help="number of choice_block layers in CNNs or operations in ViT")
parser.add_argument("--num_choices", type=int, default=4, help="number of choices per layer in CNNs or per op in ViT")
parser.add_argument('--search_num', type=int, default=10, help='the number of search iterations')
parser.add_argument('--temperature', type=float, default=0.2, help='sampling temperature')
parser.add_argument('--llm_name', type=str, default='deepseek', help='the type of llm api')

# model and dataset settings
parser.add_argument("--task", type=str, default='OfficeHome', required=True)
parser.add_argument("--data_base_path", type=str, default='~/DomainBed/domainbed/data')
parser.add_argument("--embedding_base_path", type=str, default='~/work/dgnas/embeddings')
parser.add_argument("--source", type=str, nargs='+', default=['Art'])
parser.add_argument("--target", type=str, nargs='+', default=['Clipart', 'Product', 'RealWorld'])
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--train_data_ratio", type=float, default=0.7)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)

# config file
parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

# training settings
parser.add_argument("--seed", type=int, default=0, help="training seed")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--alpha", type=float, default=0.0)
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
parser.add_argument('--mode', type=str, default='super', choices=['super', 'retrain'], help='mode of AutoFormer')

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
parser.add_argument('--mixup', type=float, default=0,
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
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, i: int, iteration_per_epoch: int
) -> None:
    warm_up = max(1, args.epochs // 10)
    base_lr = args.lr
    epochs = args.epochs
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr * T / warmup_iters
    else:
        min_lr = 1e-5
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = (
            0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr)
            + min_lr
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    aug_loader: torch.utils.data.DataLoader,
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
    train_loss = AverageMeter("TrainLoss", ":.4e")
    train_acc = AverageMeter("Acc", ":.4e")
    progress = ProgressMeter(
        len(train_loader)+len(aug_loader),
        [batch_time, curr_lr, train_loss, train_acc],
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
        # print(config)
        model_module.set_sample_config(config=config)
        # print(model_module.get_sampled_params_numel(config))

    end = time.time()
    for i, (input, target) in enumerate(aug_loader):
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

        # if mixup_fn is not None:
        #     input, target = mixup_fn(input, target)
        # print(ce_target.shape)
        loss = 0
        with torch.cuda.amp.autocast(enabled=True):
            output = model(input, is_embedding=True)
            ce_loss = criterion(output, target)
            loss += ce_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
            output = model(input, is_embedding=False)
            ce_loss = criterion(output, target)
            loss += ce_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        curr_lr.update(optimizer.param_groups[0]["lr"])
        train_loss.update(loss.item(), input.size(0))

        prec1 = accuracy(output, target, topk=(1,))[0]

        train_acc.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

    return train_loss.avg, train_acc.avg

def meta_train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    aug_loader: torch.utils.data.DataLoader,
    maml: nn.Module,
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
    adapt_loss = AverageMeter("AdaptLoss", ":.4e")
    train_loss = AverageMeter("TrainLoss", ":.4e")
    train_acc = AverageMeter("Acc", ":.4e")
    end = time.time()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, curr_lr, adapt_loss, train_loss, train_acc],
        prefix="Epoch: [{}]".format(epoch),
    )
    """
    1.break_point to decide whether augment.
    2.src_num and aug_num to decide the number of samples from source and augmented domains
    """
    num_target = len(args.target)
    break_point = int((len(train_loader) - 1) / (num_target + 1)) # numOf true iterations
    src_num = int(args.bs / (num_target + 1)) # batch_size / numOf(source_domain + target_domain) = src_num_per_batch
    aug_num = args.bs - src_num
    end = time.time()
    """
    1. We don't need generate augmentation samples from source domain samples. Actually, we already have augmented embeddings from LADS        
    2. Augmentation dataloader contains both source domains and target domains
    3. Creating train_loader and aug_loader should be done in outside of this function
    """
    aug_loader_iter = iter(aug_loader)
   
        
    maml.train()
    if args.freeze_encoder:
        if hasattr(maml, "module"):
            maml.module.encoder.eval()
        else:
            maml.encoder.eval()

    iteration_per_epoch = len(train_loader)
    
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(maml.module)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))


    for i, (input, target) in enumerate(train_loader):    
        input, target = input.cuda().float(), target.cuda().long()
        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch)
        
        # meta-train only on source domain
        optimizer.zero_grad()
        task_model = maml.clone() # torch.clone() for nn.Module

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(task_model.module)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(task_model.module)
            model_module.set_sample_config(config=config)
        # if mixup_fn is not None:
        #     input, target = mixup_fn(input, target)
        
        # # meta-test on augmented domains
        # params = list(model.parameters())
        # grads = torch.autograd.grad(loss, params, create_graph=True)
        # params = [(param - args.lr * grad).requires_grad_() for param, grad in zip(params, grads)]

        # # retrieve (1 + num_target) batch of aug_loader
        # inputs_b, targets_b = [], []
        # for _ in range(num_target):
        #     try:
        #         input_aug, target_aug = next(aug_loader_iter)
        #     except:
        #         aug_loader_iter = iter(aug_loader)
        #         input_aug, target_aug = next(aug_loader_iter)
        #     inputs_b.append(input_aug)
        #     targets_b.append(target_aug)
        # input_b, target_b = torch.cat(inputs_b, dim=0).cuda().float(), torch.cat(targets_b, dim=0).cuda().long()
        try:
            input_b, target_b = next(aug_loader_iter)
        except:
            aug_loader_iter = iter(aug_loader)
            input_b, target_b = next(aug_loader_iter)
        input_b, target_b = input_b.cuda().float(), target_b.cuda().long()
        with torch.cuda.amp.autocast(enabled=True):
            # output_b = model.functional(params, True, input_b)
            output_b = task_model(input_b, is_embedding=True) # computes loss_b using task_model
            loss_b = criterion(output_b, target_b)
        task_model.adapt(loss_b)

        with torch.cuda.amp.autocast(enabled=True):
            output = task_model(input, is_embedding=False)
            loss_a = criterion(output, target)
        # loss.backward()
        # task_model.adapt(loss_a) # computes gradients, update task_model in-place
        
        # meta-update on source domain and augmented domains, 
        # loss_combine = (loss_a + loss_b) / 2
        # optimizer.zero_grad() omit 2nd zero_grad() in l2l

        # scaler.scale(loss_combine).backward()
        # scaler.step(optimizer) # similar to optimizer.step()
        # scaler.update()
        # loss_combine.backward()

        scaler.scale(loss_a).backward()
        scaler.step(optimizer) # similar to optimizer.step()
        scaler.update()

        prec1 = accuracy(output, target, topk=(1,))[0]

        curr_lr.update(optimizer.param_groups[0]["lr"])
        adapt_loss.update(loss_a.item(), input.size(0))
        train_loss.update(loss_b.item(), input.size(0))
        train_acc.update(prec1.item(), input.size(0))
        batch_time.update(time.time() - end)

        if i % 10 == 0:
            progress.display(i)

    return train_loss.avg, train_acc.avg

@torch.no_grad()
def validate(
    val_loader: torch.utils.data.DataLoader, model: nn.Module, choices=None, mode='retrain', retrain_config=None
) -> Tuple[float, float]:

    val_loss = AverageMeter("ValLoss", ":.4e")
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

    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda().float(), target.cuda().long()
        output = model(input)

        loss = criterion(output, target)

        # if args.num_classes > 2:
        #     prec1, _ = accuracy(output, ce_target, topk=(1,5))
        # else:
        #     prec1, = accuracy(output, ce_target, topk=(1,))
        prec1 = accuracy(output, target, topk=(1,))[0]

        # bs = inputs.size(0)
        val_loss.update(loss.item(), input.size(0))
        val_acc.update(prec1.item(), input.size(0))

    return val_loss.avg, val_acc.avg

def main():
    """
    Initialize dataset, model and training settings
    """
    init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    cudnn.benchmark = True
    set_seed(args.seed)
    world_size = dist.get_world_size()
    bs = args.bs // world_size  # 256 // 2 = 128
    args.lr *= args.bs / args.bs

    # create model and load supernet weights
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
        freeze_patch_embed=True  # freeze parameters of patch_embed_super

    )
    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}
    # embed_dim = model.head.in_features
    # model.head = nn.Linear(embed_dim, args.num_classes)
    # model = Learner(model)
    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    # model = torch.compile(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()}")
    if args.mixup > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)


    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

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
        # model.load_state_dict(checkpoint["model"])
        # model.load_state_dict(checkpoint["model"]) # meta-learning
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # scaler.load_state_dict(checkpoint["scaler"])
        # best_loss = checkpoint["best_loss"]
        # best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        start_iter = checkpoint["iteration"]
    else:
        start_epoch = 0
        start_iter = 0

    # # integrate model into MAML algorithm for meta-learning
    # maml = MAML(model, lr=args.lr*100, allow_nograd=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     weight_decay=args.wd,
    # )

    """
    1.Load train_dataset & augment_dataset to finetune for domain adaptation
        1.1 augment_dataset in the form of CLIP embeddings
    2.Load val_dataset to rank and search the optimal architecture
    3.Load target_dataset to evaluate the performance of the finetuned model
    """
    train_dataset, val_dataset, _ = get_source_dataset(args)
    _, augmented_dataset = get_augmented_dataset(args)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(augmented_dataset))

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
    aug_sampler = torch.utils.data.distributed.DistributedSampler(augmented_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    aug_loader = torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size=bs,
        # shuffle=(aug_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=aug_sampler,
        drop_last=True,
        persistent_workers=True,
    )

    if args.train_full:
        val_loader = None
    else:
        _, _, target_dataset = get_target_dataset(args)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        target_sampler = torch.utils.data.distributed.DistributedSampler(target_dataset, num_replicas=world_size, rank=rank, shuffle=False)
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
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=bs,
            # shuffle=(target_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=target_sampler,
            drop_last=False,
            persistent_workers=True,
        )
    """
    Initialize the search index conversion and search history storage
    """
    # num_heads_map = {'3':0, '4':1, '6':2, '8':3}
    # mlp_ratio_map = {'3':0, '4':1, '5':2}
    # depth_map = {'12':0, '13':1, '14':2}

    arch_list = []
    acc_list = []
    performance_history = []
    messages_history = []

    if not os.path.exists('history'):
        os.makedirs('history')
    
    # system_content = "You are an expert in the field of neural architecture search."
    with open("search/search_prompt.json", "r") as f:
        search_prompt = json.loads(f.read())
    system_content = search_prompt['vit4cls']['system_content']
    user_input = search_prompt['vit4cls']['user_input']  # Note for this, load search_prompt as a json file

    experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
    {}
    Please suggest another set of model config that can improve the model's performance on target domains beyond the experimental results provided above. You can provide another set of network architecture configs different from the previous round(including depth, a list of num_heads, a list of mlp ratios).
    '''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))
    domain_input = "The source domain is {args.source}, and the target domain is {args.target}."
    suffix = search_prompt['vit4cls']['suffix']
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input + domain_input + suffix},
    ]

    """
    Initialize the search process with llm apis(GPT-4o/Qwen/Llama)
    """
    # Initialize openai settings
    # openai.api_key = args.openai_key
    # openai.organization = args.openai_organization    

    def get_llm_response(messages):
        client = OpenAI(api_key=llm_params[args.llm_name]['api_key'], base_url=llm_params[args.llm_name]['base_url'])
        res = client.chat.completions.create(model=llm_params[args.llm_name]['name'], messages=messages, stream=False, temperature=args.temperature, n=1).choices[0].message  # request prompt and response
        messages.append({"role": res.role, "content": res.content})
        messages_history.append(messages)

        print(repr(res.content))

        depth = json.loads(res.content.split('\n')[0])
        num_heads = json.loads(res.content.split('\n')[1])
        mlp_ratio = json.loads(res.content.split('\n')[2])
        # embed_dim = json.loads(res.content.split('\n')[3])
        return depth, num_heads, mlp_ratio


    for iteration in range(start_iter, args.search_num):
        # we should provide a unified interface for calling various llm api
        depth, num_heads, mlp_ratio = get_llm_response(messages)
        # depth = 13
        # num_heads = [6, 5, 5, 7, 5, 7, 7, 7, 7, 7, 5, 5, 6]
        # mlp_ratio = [3.0, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 3.0, 4.0, 3.0, 3.5]
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [384]*depth


        best_loss = 1e4
        best_acc = 0.0

        # # calculate FLOPs
        # input = torch.randn(1, 3, args.img_size, args.img_size)
        # if args.model == 'cnn':
        #     choice = kernels
        #     macs, params = profile(model, inputs=(input,choice))
        # else:
        #     depth_idx = gpt_depth
        #     attn_idx = gpt_heads
        #     mlp_idx = gpt_mlp_ratios
        #     macs, params = profile(model, inputs=(input,depth_idx, attn_idx, mlp_idx))
        #     print("Flops: {}, Params: {}".format(macs / 1e9, params / 1e6))
        #     flops = macs / 1e9

        # Train and validation process
        if args.eval_only:
            # if eval_only, only evaluate on target domain data
            val_loss, val_acc = validate(val_loader=target_loader, model=model_without_ddp, mode='retrain', retrain_config=sampled_config)
            print(
                "Epoch: {} val_loss {:.4f} - val_acc {:.4f}".format(
                    start_epoch - 1, val_loss, val_acc
                )
            )
        else:
            for epoch in range(0, args.epochs):
                train_sampler.set_epoch(epoch)
                _, _ = train_one_epoch(train_loader, aug_loader, model, criterion, optimizer, scaler, epoch, mixup_fn, mode='retrain', retrain_config=sampled_config)
                # scheduler.step()
                if val_loader is not None:
                    val_loss, val_acc = validate(val_loader=val_loader, model=model_without_ddp, mode='retrain', retrain_config=sampled_config)
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
                        "iteration": iteration + 1,
                    }
                    torch.save(state_dict, checkpoint_path)
            # final evaluation on target domain data after each search iteration
            test_loss, test_acc = validate(val_loader=target_loader, model=model_without_ddp, mode='retrain', retrain_config=sampled_config)
            print(
                "Epoch: {} test_loss {:.4f} - test_acc {:.4f}".format(
                    args.epochs - 1, test_loss, test_acc
                )
            )

        # Return with validation accuracy
        accuracy = float(Decimal(best_acc).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))

        arch_list.append({'depth':depth, 'num_heads':num_heads, 'mlp_ratio':mlp_ratio, 'embed_dim':384})
        acc_list.append(accuracy)
        rank = sorted(acc_list,reverse=True).index(accuracy) + 1

        performance = {
            'arch' : {'depth':depth, 'num_heads':num_heads, 'mlp_ratio':mlp_ratio, 'embed_dim':384},
            'rank' : "{}/{}".format(rank, len(acc_list)),
            'acc'  : str(accuracy),
            # 'flops': str(flops),
        }

        print(iteration+1, performance)

        performance_history.append(performance)


        messages_file = 'history/vit_res_messages.json'
        performance_file = 'history/vit_res_performance.json'
        with open(messages_file, 'w') as f:
            json.dump(messages_history, f)

        with open(performance_file, 'w') as f:
            json.dump(performance_history, f)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
        ]  # used for next iteration
        # load pretrained model for next search iteration
        if args.finetune_ckpt is not None:
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
    destroy_process_group()

if __name__ == '__main__':
    main()