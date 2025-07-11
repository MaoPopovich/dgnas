import torch.utils.data as data
from PIL import Image
import lmdb
import pickle
import io
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import sys
import os
sys.path.append(os.getcwd())
from data.augmentation import *
from typing import List,Tuple
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import pandas as pd
from data.pacs_utils import prepare_PACS, prepare_PACS_embedding
from data.vlcs_utils import prepare_VLCS, prepare_VLCS_embedding
from data.domainnet_utils import prepare_domainnet, prepare_domainnet_embedding
from data.officehome_utils import prepare_OH, prepare_OH_embedding
from torch.utils.data import ConcatDataset

"""
PACS, 4 domains and 7 classes
VLCS, 4 domains and 5 classes
OfficeHome, 4 domains and 65 classes(mini to 10 classes)
DomainNet, 6 domains and 345 classes(mini to 10 classes)
"""
vlcs_datasets = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
pacs_datasets = ['photo', 'art_painting', 'cartoon', 'sketch']
officehome_datasets = ['Art', 'Clipart', 'Product', 'RealWorld']
domainnet_datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
domain_map = {'PACS': pacs_datasets, 'VLCS': vlcs_datasets, 'OfficeHome': officehome_datasets, 'DomainNet': domainnet_datasets}

class TinyImagenet(data.Dataset):
    def __init__(self, dataset_path='diabetic/train', transforms=None):
        # # Normalization
        # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        #     if pretrained else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # # Normal transformation
        # if pretrained:
        #     train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), 
        #                     transforms.ToTensor()]
        #     val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
        # else:
        #     train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        #     val_trans = [transforms.ToTensor(), norm]
        
        # trans = train_trans + [norm] if enable_train else val_trans

        self.dataset = datasets.ImageFolder(dataset_path, transform=transforms)

    def __len__(self):
            return self.dataset.__len__()
    def __getitem__(self, index):
            return self.dataset.__getitem__(index)

@torch.no_grad()
def batch_process_label(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    class0 = label == 0
    class1 = label == 1
    stack_label = torch.stack([class0, class1], dim=1)   # label for ce_loss, stack_label for dice loss
    stack_label = (stack_label > 0.5).float()
    return label.long(), stack_label

# 动态应用transform
def apply_transform(dataset, transform):
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform=None):
            super().__init__()
            self.base_dataset = base_dataset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.base_dataset[index]
            if self.transform is not None:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.base_dataset)
    
    return TransformedDataset(dataset, transform)

def init_imagenet(data_dir='./dataset/tiny-imagenet-1k/train', pretrained=True, args=None):
    dataset = datasets.ImageFolder(root=data_dir)
    
    if not pretrained:
        return dataset

    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(0.9 * len(dataset))
    train_dataset = data.Subset(dataset, indices[:train_size])
    val_dataset = data.Subset(dataset, indices[train_size:])

    # 应用transform到各自的子集
    train_dataset = apply_transform(train_dataset, train_transform)
    val_dataset = apply_transform(val_dataset, val_transform)

    return train_dataset, val_dataset

def build_transform(is_train, args):
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
        # replace RandomResizedCropAndInterpolation with
        # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.img_size, padding=4)
        return transform
    
    t = []
    if resize_im:
        size = int((224 / 224) * args.img_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_mask_transform(args):
    mask_transforms = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), 
                                      transforms.PILToTensor()])
    return mask_transforms


def get_source_dataset(args):
    source_domain = args.source
    train_data = []
    val_data = []
    source_data = []

    if args.task == 'PACS':
        train_datasets, val_datasets, full_datasets = prepare_PACS(args.data_base_path, args.train_data_ratio)
    elif args.task == 'VLCS':
        train_datasets, val_datasets, full_datasets = prepare_VLCS(args.data_base_path, args.train_data_ratio)
    elif args.task == 'OfficeHome':
        train_datasets, val_datasets, full_datasets = prepare_OH(args.data_base_path, args.train_data_ratio)
    elif args.task == 'DomainNet':
        train_datasets, val_datasets, full_datasets = prepare_domainnet(args.data_base_path, args.train_data_ratio)
    else:
        raise NotImplementedError('DATA LOADER NOT IMPLEMENTED.')
    # choose data from source domains
    if isinstance(source_domain, list):
        for i, domain in enumerate(source_domain):
            index = domain_map[args.task].index(domain)
            train_data.append(train_datasets[index])
            val_data.append(val_datasets[index])
            source_data.append(full_datasets[index])
    else:
        index = domain_map[args.task].index(source_domain)
        train_data.append(train_datasets[index])
        val_data.append(val_datasets[index])
        source_data.append(full_datasets[index])
    
    train_dataset = ConcatDataset(train_data)
    val_dataset = ConcatDataset(val_data)
    source_dataset = ConcatDataset(source_data)
    # train_loader = torch.utils.data.DataLoader(datasets, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_dataset, val_dataset, source_dataset

def get_target_dataset(args):
    target_domain = args.target
    assert isinstance(target_domain, list)
    train_data = []
    val_data = []
    target_data = []

    if args.task == 'PACS':
        train_datasets, val_datasets, full_datasets = prepare_PACS(args.data_base_path, args.train_data_ratio)
    elif args.task == 'VLCS':
        train_datasets, val_datasets, full_datasets = prepare_VLCS(args.data_base_path, args.train_data_ratio)
    elif args.task == 'OfficeHome':
        train_datasets, val_datasets, full_datasets = prepare_OH(args.data_base_path, args.train_data_ratio)
    elif args.task == 'DomainNet':
        train_datasets, val_datasets, full_datasets = prepare_domainnet(args.data_base_path, args.train_data_ratio)
    else:
        raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')
    
    
    for i, domain in enumerate(target_domain):
        index = domain_map[args.task].index(domain)
        train_data.append(train_datasets[index])
        val_data.append(val_datasets[index])
        target_data.append(full_datasets[index])
    
    train_dataset = ConcatDataset(train_data)
    val_dataset = ConcatDataset(val_data)
    target_dataset = ConcatDataset(target_data)
    # val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
    return train_dataset, val_dataset, target_dataset

def get_augmented_dataset(args):
    source_domain = args.source
    if isinstance(source_domain, list):
        source_domain = source_domain[0]
    
    if args.task == 'PACS':
        augmented_dataset = prepare_PACS_embedding(args.embedding_base_path, source_domain)
    elif args.task == 'VLCS':
        augmented_dataset = prepare_VLCS_embedding(args.embedding_base_path, source_domain)
    elif args.task == 'OfficeHome':
        augmented_dataset = prepare_OH_embedding(args.embedding_base_path, source_domain)
    elif args.task == 'DomainNet':
        augmented_dataset = prepare_domainnet_embedding(args.embedding_base_path, source_domain)
    else:
        raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')

    return augmented_dataset

       
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_path', type=str, default='/home/qinghua_mao/work/DomainBed/domainbed/data')
    parser.add_argument('--task', type=str, default='DomainNet')
    parser.add_argument('--source', type=str, nargs='+', default=['clipart', 'infograph', 'painting', 'quickdraw'])
    parser.add_argument('--target', type=str, default=['real', 'sketch'])
    parser.add_argument('--train_data_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train_dataset, val_dataset = get_source_dataset(args)

    
    