<div align="center">

# Towards Domain Generalized Neural Architecture Search via Language-Guided Visual Augmentation

[![Conference](https://img.shields.io/badge/ACM%20MM-2026-b31b1b.svg)](https://2026.acmmm.org/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

**Official PyTorch implementation of DGNAS (ACM Multimedia 2026)**

**Qinghua Mao**<sup>1</sup>, **Xi Lin**<sup>1,&dagger;</sup>, **Zhongze Wu**<sup>2</sup>, **Jun Wu**<sup>1</sup>, **Xiangrui Cai**<sup>3</sup>, **Xiu Su**<sup>2</sup>

<sup>1</sup>Shanghai Jiao Tong University &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup>Central South University &nbsp;&nbsp;|&nbsp;&nbsp; <sup>3</sup>Nankai University  
<sup>&dagger;</sup>Corresponding author: `linxi234@sjtu.edu.cn`

</div>

---

## 📢 News
- **[2026.07]** 🎉 Our paper **"Towards Domain Generalized Neural Architecture Search via Language-Guided Visual Augmentation"** has been accepted by **ACM Multimedia 2026 (ACM MM '26)**!
- **[2026.09]** 🚀 Codebase, search scripts, and pretrained supernet weights are released.

---

## 💡 Overview

Conventional Neural Architecture Search (NAS) relies on the in-distribution (I.I.D.) assumption, searching subnets on source validation data and inevitably discovering architectures that overfit source statistics. Meanwhile, existing Domain Generalization (DG) methods keep deep backbones static, neglecting the critical role of architecture scaling in out-of-distribution robustness.

**DGNAS** pioneers the **"Augment First, Search and Fine-Tune"** paradigm:
1. **Language-Guided Visual Augmentation**: Synthesizes realistic OoD domains in the joint vision-language space (CLIP) using verbal descriptions alone, optimizing both Directional Adaptation ($\mathcal{L}_{\text{DA}}$) and Semantic Consistency ($\mathcal{L}_{\text{SC}}$).
2. **LLM-Driven Architecture Navigation**: Leverages an LLM reasoning agent equipped with Domain-Aware NAS Prompts (DNP) and Pareto memory to actively propose optimal subnets in **only 10 rounds**.
3. **Meta Adaptation Fine-Tuning**: Employs bi-level gradient adaptation across source and synthesized auxiliary domains to ensure swift generalization to unseen domains.

<div align="center">
  <img src="figure/framework.pdf" width="95%" alt="DGNAS Framework">
  <p><em>Figure 1: Overview of the DGNAS framework. (a) The "Augment first, search and fine-tune" paradigm. (b) Detailed pipeline of language-guided visual augmentation and LLM-driven architecture search.</em></p>
</div>

---

## 🏆 Main Results

### 1. Comparison with SOTA DG & NAS Methods (Top-1 Accuracy %)

| Method | Type | PACS | VLCS | OfficeHome | DomainNet |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ERM** | Standard Baseline | 84.20 | 77.30 | 67.60 | 44.00 |
| **DomainDrop** | DG Alignment | 87.90 | 79.80 | 68.70 | 44.40 |
| **PromptStyler** | VLM Augmentation | 93.20 | 82.30 | 73.60 | 49.50 |
| **SPG** | VLM SOTA | 97.00 | 82.40 | 83.60 | 60.10 |
| **AutoFormer** | Vision NAS | 92.83 | 69.65 | 82.54 | 85.74 |
| **DGNAS (Ours)** | **VLM-Guided NAS** | **97.64** | **84.30** | **99.70** | **96.12** |

> **Key Takeaway**: DGNAS establishes new SOTA records across all 4 benchmarks, outperforming top VLM-augmented DG methods by **+36.02%** on DomainNet and standard NAS (AutoFormer) by **+14.65%** on VLCS.

### 2. Extreme Search Efficiency
- **Search Time**: Achieves SOTA Pareto ViT architectures in **3.20 ~ 4.88 GPU hours** (vs. AutoFormer 10.5h, TF-TAS 30.0h).
- **Fast Convergence**: LLM reasoning navigates discrete architecture space to Pareto optimality in **only 10 conversational iterations**.

---

## 📂 Repository Structure

```text
dgnas/
├── backbone/                  # Supernet ViT & dynamic building blocks
│   ├── supernet_transformer.py
│   └── module/               # Super multi-head attention, Linear & LayerNorm
├── data/                      # Multi-domain dataset loaders
│   ├── pacs_utils.py
│   ├── vlcs_utils.py
│   ├── officehome_utils.py
│   └── domainnet_utils.py
├── experiments/               # Supernet & searched subnet configurations
│   ├── supernet/             # Supernet-T / Supernet-S / Supernet-B yaml configs
│   └── subnet/               # Searched subnet yaml configs for PACS/VLCS/OH/DN
├── search/                    # LLM domain-aware prompting & reasoning utils
│   ├── search_prompt.py
│   └── test_prompt.py
├── script/                    # Shell automation pipelines
│   ├── command.sh
│   ├── train.sh              # Supernet pretraining script
│   └── search.sh             # DGNAS search & retraining pipeline
├── train_cls_superformer.py   # Supernet pretraining & subnet retraining entry
├── gpt_search_finetune_former.py # LLM-driven search & fine-tuning entry
└── environment.yaml           # Conda environment dependency specification
```

---

## 🛠️ Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/MaoPopovich/dgnas.git
cd dgnas

# Create and activate conda environment
conda env create -f environment.yaml
conda activate dgnas

# Configure your OpenAI API Key for the LLM Search Agent
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Dataset Preparation

Please download the following DG benchmark datasets and organize them into the `data/` directory:
- [PACS](https://domaingeneralization.github.io/) (Photo, Art Painting, Cartoon, Sketch)
- [VLCS](https://domaingeneralization.github.io/) (Caltech, LabelMe, Pascal, Sun)
- [OfficeHome](https://hemanthdv.github.io/officehome-dataset/) (Art, Clipart, Product, RealWorld)
- [DomainNet](http://ai.bu.edu/M3SDA/) (Clipart, Infograph, Painting, Quickdraw, Real, Sketch)

---

## ⚡ Execution Pipeline

### Step 1: Pre-training the ViT Supernet
Pre-train the weight-sharing ViT supernet on ImageNet-1K:
```bash
bash script/train.sh
```

### Step 2: Language-Guided Augmentation & LLM-Driven Search
Run LLM-driven architecture search on target benchmarks (e.g., PACS, VLCS, OfficeHome, DomainNet):
```bash
# Search on PACS (Source: photo -> Targets: art_painting, cartoon, sketch)
bash script/command.sh ft-s0-vit-pacs \
  "torchrun --standalone --nnodes=1 --nproc_per_node=2 gpt_search_finetune_former.py \
   --split 0 --model vit --task PACS --source photo --target art_painting cartoon sketch \
   --num_classes 7 --alpha 0.0 --wd 1e-4 --epochs 50 --lr 5e-3 --bs 32 --img_size 224 \
   --arch vit_entangle_patch16_224 --cfg experiments/supernet/supernet-S.yaml \
   --checkpoint ft-s0-vit-pacs --finetune_ckpt supernet-small"
```

### Step 3: Retraining the Searched Subnet
Retrain the optimal subnet discovered by DGNAS from scratch:
```bash
bash script/command.sh rt-s0-vit-pacs \
  "torchrun --standalone --nnodes=1 --nproc_per_node=2 train_cls_superformer.py \
   --split 0 --model vit --task PACS --source photo --target art_painting cartoon sketch \
   --num_classes 7 --alpha 0.0 --wd 1e-4 --epochs 200 --lr 5e-3 --bs 32 --img_size 224 \
   --arch vit_entangle_patch16_224 --cfg experiments/subnet/PACS/AutoFormer-S.yaml \
   --checkpoint rt-s0-vit-pacs --finetune_ckpt supernet-small"
```

---

## 📚 Citation

If you find our paper or codebase helpful in your research, please consider citing:

```bibtex
@inproceedings{mao2026dgnas,
  author    = {Mao, Qinghua and Lin, Xi and Wu, Zhongze and Wu, Jun and Cai, Xiangrui and Su, Xiu},
  title     = {Towards Domain Generalized Neural Architecture Search via Language-Guided Visual Augmentation},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  pages     = {1--14},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

---

## 📄 License
This project is released under the [Apache 2.0 License](LICENSE).

## 📬 Contact
For questions or collaborations, please feel free to contact **Qinghua Mao** (`mmmm2018@sjtu.edu.cn`) or **Xi Lin** (`linxi234@sjtu.edu.cn`).
