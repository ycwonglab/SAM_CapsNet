# SAM-CapsNet

**Attention-Conditioned Capsule Networks: A Multi-Scale Attention Module (SAM-CapsNet) for Video-Based Human Activity Recognition**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23254308.svg)](https://doi.org/10.5281/zenodo.23254308)

This repository is the replication record for the manuscript. It holds:

- [`SAM_CapsNet_Reproducibility_FINAL.ipynb`](SAM_CapsNet_Reproducibility_FINAL.ipynb), the notebook that defines the models and training routines and produces the tables and figures of the paper;
- [`splits/`](splits/), the train / validation / test file lists for every seed on UCF-11 and UCF-101.

The notebook is archived on Zenodo at **[10.5281/zenodo.23254308](https://doi.org/10.5281/zenodo.23254308)** (this DOI always points to the latest version).

> **The manuscript is under review, so no accuracy values are published in this repository.** The
> notebook here has its outputs cleared. Running it regenerates the tables and figures from your
> own copies of UCF-11 and UCF-101.

Two earlier notebooks are kept in [`archive/`](archive/) for reference only. They do not match the current manuscript.

## Model

A shared four-layer convolutional encoder turns each 112 x 112 frame into a 256-channel 7 x 7 feature map. Each frame is refined by the attention module, the 16 frames are averaged and layer-normalised, and a capsule stage classifies the clip.

The attention module applies four steps in a fixed order:

1. **Channel attention.** Average and max pooling share one bottleneck MLP (reduction ratio 8); the result is applied multiplicatively.
2. **Motion map.** Frame `t` is concatenated with frame `t-1` and passed through parallel 3x3 and 5x5 convolutions, fused by a 1x1 convolution. The first frame has no motion map.
3. **Spatial attention.** A learnable 256-dimensional vector (the point prompt) is added to the input of this branch, which is combined with the motion map and reduced to a spatial map that is applied multiplicatively.
4. **Motion residual.** `out + tanh(alpha) * (out * a_m)`, with `alpha` initialised at 0.1.

The fusion weight `lam` is used only to draw attention maps. It is not part of the forward pass.

The capsule stage forms 16 primary-capsule types of dimension 8 at each of the 7 x 7 positions (784 primary capsules) and runs three routing iterations to one 16-dimensional activity capsule per class. The class score is the capsule length.

## Training protocol

All settings are in **Section 7** of the notebook.

| | UCF-11 | UCF-101 |
|---|---|---|
| Classes | 11 | 101 |
| Input | 112 x 112, 16 frames at a fixed 5 fps | same |
| Split | stratified 70 / 15 / 15 at clip level, one per seed | same |
| Clips per split | 1,119 / 240 / 241 | 9,323 / 1,998 / 1,999 |
| Epochs | 15 | 15 |
| Batch | 4, gradient accumulation 4 | 2, gradient accumulation 4 |

- **Optimiser.** Adam, learning rate 1e-3, weight decay 1e-4, three-epoch linear warm-up then cosine decay, gradient clipping at 5.0.
- **Seeds.** 42, 123, 456, 789 and 2024. A seed fixes the split and all random number generators.
- **Video order.** Videos are listed in sorted order, so each split depends only on its seed.
- **Model selection.** The checkpoint with the best validation accuracy is evaluated on the test set.
- **No data augmentation.**

Two training routines are used, as stated in Section II-F of the paper:

- **Margin-loss routine** (Sections 19 to 23 and 27). The margin loss of Sabour et al. (2017) on the capsule lengths. Used for Tables 1, 2 and 4 and the capsule rows of Table 3.
- **Cross-entropy routine** (Sections 9, 11, 16 and 18). Cross-entropy on the capsule lengths with mixed precision. Used for the attention-placement study, the SAM-CapsNet confusion matrix, the orientation test and the single-clip figures.

On UCF-101 the routing weights of the Sabour-style baseline are scaled by K/11 at initialisation (Section 27). The ImageNet-pretrained ResNet-18 and MobileNetV2 baselines are trained at learning rate 1e-3 (Section 25) and at a fine-tuning learning rate of 1e-4 (Section 26).

The split is at clip level. It does not keep the 25 groups of UCF-11 apart, so clips from the same group can fall in both the training and test sets.

## Which section produces which result

| Paper item | Notebook section |
|---|---|
| Supplementary Table S3 (class counts); split file lists | 8 |
| Table 1, CNN rows | 10 |
| Sabour-style baseline under cross-entropy (Section II-E) | 11 |
| Figure 5 | 12 |
| Figure 2 | 13 |
| Figure 3 and Supplementary Figure S1 | 14 |
| Figure 4 | 15 |
| Table 5 (orientation test) | 16 |
| Inference latency (Section III-F) | 17 |
| Attention-placement study (Section III-A) | 18 |
| Table 1, capsule rows; Table 4 (routing concentration) | 19 |
| Table 1 and Table 4, network with attention removed | 20 |
| Table 3, pre-trained ResNet-18 stem on UCF-11 | 21 |
| Supplementary Table S2 | 22 |
| Significance tests (Section III-A); Table 2 (per-class accuracy) | 23 |
| Table 3, ResNet-18 and MobileNetV2 on UCF-101 | 25 (learning rate 1e-3), 26 (learning rate 1e-4) |
| Table 3 and Section III-F, capsule models on UCF-101 | 27 and 27B |

## Split file lists

`splits/ucf11_seed<seed>.csv` and `splits/ucf101_seed<seed>.csv` have three columns: `split` (train, val or test), `class` and `video` (path relative to the dataset root). Section 8 and Section 24 of the notebook write the same lists to `results_final/splits/`.

## Running it

Requires a CUDA GPU, PyTorch with torchvision, and local copies of UCF-11 and UCF-101.

1. Set `DATASET_PATH` (UCF-11) and `UCF101_PATH` (UCF-101) in Section 7. Each dataset root holds one folder per class.
2. For a fast check, set `QUICK_TEST = True` and choose *Restart & Run All*. Every section then uses one seed and two epochs. The numbers are not meaningful.
3. Set `QUICK_TEST = False` and run all cells for the full run. Results, figures and split lists are written to `results_final/`.

The UCF-101 sections (24 to 27) come last and take the longest. Set `RUN_UCF101 = False` to stop before them. The first run builds frame caches in `.cache_ucf11/` and `.cache_ucf101_r1/`.

### Datasets

- UCF-11 (YouTube Action): https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php

## Not in this repository

- **Accuracy values, logs and checkpoints.** Withheld while the manuscript is under review.
- **The datasets.** UCF-11 and UCF-101 are distributed by their original authors at the links above.

## Availability

This record is unlisted rather than secret. It is reachable by direct URL for reviewers and editors, and `robots.txt` together with a `noindex` directive keeps it out of search results while the manuscript is under review. Both will be removed on acceptance.
