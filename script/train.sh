#!/bin/bash

# train supernet with vit on ImageNet
bash script/command.sh pt-s0-vit-base    "torchrun --standalone --nnodes=1 --nproc_per_node=2 train_cls_supernet.py --split 0     --model vit --num_classes 1000 --alpha 0.0 --wd 5e-2 --epochs 500 --lr 0.001 --bs 512 --img_size 224 --arch vit_base_patch16_224  --checkpoint pt-s0-vit-base"
bash script/command.sh pt-s0-vit-auto    "torchrun --standalone --nnodes=1 --nproc_per_node=2 train_cls_superformer.py --split 0     --model vit --num_classes 1000 --alpha 0.0 --wd 5e-2 --epochs 500 --lr 0.001 --bs 512 --img_size 224 --arch vit_entangle_patch16_224 --cfg experiments/supernet/supernet-B.yaml --checkpoint pt-s0-vit-auto"
