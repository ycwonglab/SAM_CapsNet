# SAM-CapsNet

**Capsule Network with Multi-Scale Attention for Video-Based Human Activity Recognition**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23254309.svg)](https://doi.org/10.5281/zenodo.23254309)

This repository is the replication record for the manuscript. Its reference notebook,
[`SAM_CapsNet_Complete_Reproducibility.ipynb`](SAM_CapsNet_Complete_Reproducibility.ipynb),
defines the model, the ablation grid, the baselines and the analysis code for every
experiment, table and figure in the paper. An archived copy is on Zenodo at
**[10.5281/zenodo.23254309](https://doi.org/10.5281/zenodo.23254309)**. To cite the
notebook, use this DOI.

> **The manuscript is under review, so no accuracy values are published here.** The notebook is
> distributed with its outputs cleared. Running it end to end regenerates the tables and figures
> from your own copies of UCF-11 and UCF-101.

The earlier replication notebook,
[`Capsule_Network_with_Multi_Scale_Attention_(SAM_CapsNet)_for_Video_Based_Human_Activity_Recognition.ipynb`](Capsule_Network_with_Multi_Scale_Attention_(SAM_CapsNet)_for_Video_Based_Human_Activity_Recognition.ipynb),
is kept for reference. Everything below describes the complete reproducibility notebook.

## Architecture

A shared four-layer convolutional stem encodes every frame independently. The stem is a 7x7
stride-2 convolution to 64 channels, then 3x3 convolutions to 128, 256 and 256 channels, with
three max-pooling stages. A 112 x 112 frame becomes a 256-channel 7 x 7 feature map. Each frame
is refined by the Multi-Scale SAM block, and the refined frames are mean-pooled over time and
layer-normalised before the capsule stage.

The attention block applies four steps in a fixed order:

1. **Channel.** CBAM-style squeeze-excitation. Average and max pooling share one bottleneck MLP
   (reduction ratio 8), summed and passed through a sigmoid, applied multiplicatively.
2. **Temporal motion map.** Frame `t` is concatenated with frame `t-1` and passed through two
   parallel convolutions at different receptive fields (3x3 and 5x5), fused by a 1x1 convolution
   and a sigmoid. Frame 0 has no predecessor and so receives no motion map.
3. **Spatial.** A learned point prompt (initialised from N(0, 0.01)) is added to the
   channel-refined features. These are concatenated with the motion map, projected by a 1x1
   guide convolution and passed through a 1x1 / 3x3 / 1x1 convolution stack ending in a
   sigmoid, applied multiplicatively.
4. **Temporal residual.** `out + tanh(alpha) * (out * a_m)`, with `alpha` initialised at 0.1. The
   `tanh` gate is signed, so motion regions can be amplified rather than only suppressed.

The fusion weight `lam` (initialised at 0.5) blends the spatial and motion maps into the single
attention map that the visualisation cells (Sections 13 to 16) draw. It is not part of the
training forward pass in this version of the notebook.

The capsule stage is a standard two-layer arrangement. `PrimaryCaps` is a 3x3 convolution
reshaped into 16 capsule types of 8 dimensions at each of the 7 x 7 positions (784 input
capsules) and squashed. `ActivityCaps` runs three dynamic-routing iterations to one
16-dimensional capsule per class. The class decision is the L2 norm of the output capsules,
trained with cross-entropy on those norms.

## Training protocol

All settings live in **Section 7 (Configuration)**.

| | UCF-11 (Sections 9 to 16, 18 to 23) | UCF-101 (Section 17) |
|---|---|---|
| Classes | 11 | 101 |
| Input resolution | 112 x 112 | 112 x 112 |
| Frames per clip | 16, sampled at a fixed 5 fps | 16, sampled at a fixed 5 fps |
| Split | 70 / 15 / 15 stratified, re-drawn per seed | 70 / 15 / 15 stratified, re-drawn per seed |
| Epochs | 30 (`NUM_EPOCHS`) | 30 (`NUM_EPOCHS`) |
| Batch | 4 x gradient accumulation 4 (effective 16) | 2 x gradient accumulation 4 (effective 8) |

Every SAM-CapsNet and Pure CapsNet run uses Adam at learning rate 1e-3 with weight decay 1e-4, a
3-epoch linear warm-up followed by cosine decay, gradient clipping at 5.0, mixed precision on
CUDA, and seeds 42 / 123 / 456 / 789 / 2024. Frames are resized and ImageNet-normalised with no
training-time augmentation. The checkpoint with the best validation accuracy is the one evaluated
on test.

The UCF-101 baselines, ResNet-18 and MobileNetV2, start from ImageNet-pretrained torchvision
weights, mean-pool per-frame features over time and use a linear classifier. They share the
split, frame count, resolution, epochs and seeds. Their learning rate halves every 7 epochs.

The two CNN rows of Table 1 come from `BaselineCNN` (Section 4B), which reuses the convolutional
stem but replaces the capsule stage with mean- and max-pooled temporal aggregation and an MLP
softmax classifier, with its channel attention off (Baseline CNN) or on (CNN + SAM). Section 9B
trains both variants at seed 42 on a 70 / 15 / 15 split, with Adam at learning rate 5e-4 and
weight decay 1e-4.

The rotation-adaptation experiment (Section 21) trains for 15 epochs (`RR_EPOCHS`).

## Ablation grid

Eight capsule configurations, registered in Section 5 and run by Section 10.

| Key | Label | Channel | Spatial | Temporal | Point prompt | Learnable fusion |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `full_sam` | SAM-CapsNet (proposed) | yes | yes | yes | yes | yes |
| `no_sam` | Pure CapsNet | no | no | no | no | no |
| `ch_only` | Channel only | yes | no | no | no | no |
| `tm_only` | Temporal only | no | no | yes | no | no |
| `sp_only` | Spatial only | no | yes | no | no | no |
| `ch_tm` | Channel + Temporal | yes | no | yes | no | no |
| `tm_sp` | Temporal + Spatial | no | yes | yes | no | no |
| `ch_sp` | Channel + Spatial | yes | yes | no | no | no |

Note that `full_sam` is the only configuration carrying the point prompt, so the step from
`tm_sp` or `ch_sp` to the proposed model moves more than one component at a time.

## Which section produces which result

Section numbers are the numbered headings in the notebook.

| Section | Produces |
|---|---|
| 1 to 8 | Setup, dataset, model, ablation registry, configuration, dataset load |
| 4B | `BaselineCNN` definition and its training and evaluation helpers |
| 9 | Quick single-seed check (seed 42) |
| 9B | Table 1 rows 1 and 2 (Baseline CNN, CNN + SAM), and the model for Figure 7 (a) |
| 10 | Table 1 capsule rows, Figures 3 and 4 (8 configurations x 5 seeds) |
| 11 | Figure 7 (confusion matrices) |
| 12 | Table 2 (per-class accuracy) |
| 13 | Figure 2 (activity capsule magnitudes and angles) |
| 14 | Figure 5 (b, c, d) (capsule vector analysis) |
| 15 | Figure 5 (a) (attention with centre-of-mass overlay) |
| 16 | Figure 6 (per-class capsule distributions) |
| 17 | Table 3, Section A (UCF-101: SAM-CapsNet, ResNet-18, MobileNetV2, with parameter counts) |
| 18 | Tables S2 and S3 (capsule routing concentration) |
| 19 | Table 4, Figure 10 (capsule orientation equivariance, horizontal flip and 90 degree rotation) |
| 20 | Table 5, Figure 8 (viewpoint robustness, no retraining) |
| 21 | Table 6, Figure 9 (rotation-adapted training on flipped clips) |
| 22 | Per-clip inference latency |
| 23 | Subject-independent (leave-groups-out) split |

The CNN+LSTM entry in Table 3, Section C, is cited from the literature and not trained here.

## Running it

Requires a CUDA GPU, PyTorch with torchvision, and local copies of UCF-11 and UCF-101.

1. Set `DATASET_PATH` (UCF-11) and `UCF101_PATH` (UCF-101) in **Section 7**. Each dataset root
   should hold one folder per class.
2. Run Sections 1 to 8. Section 8 builds the frame cache in `.cache_ucf11/`. The first pass
   decodes video and is slow; later passes load from the cache.
3. Run Section 9 as a single-seed smoke test before starting the full grid.
4. Run Section 9B for the CNN baselines, then Sections 10 to 16 for Tables 1 and 2 and
   Figures 2 to 7. Section 11 needs the output of both 9B and 10.
5. Run Section 17 for the UCF-101 benchmark. It caches frames in `.cache_ucf101_r1/`.
6. Run Sections 18 to 23 for the supplementary analyses. Section 20 reuses the models kept in
   memory by Section 10, and Section 22 times the Section 9 model.

Results are written to `results/` (`SAVE_DIR`) as JSON and PNG files, for example
`ablation_table.json`, `ucf101_benchmark.json`, `equivariance_test.json` and
`fig7_confusion_matrices.png`.

### Datasets

- UCF-11 (YouTube Action): https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php

The subject-independent splitter in Section 23 reads group identifiers from either the filename
(`v_Basketball_g01_c01`) or the parent folder (`Basketball/group01/`).

## Deliberately not here

- **Accuracy values, logs and checkpoints.** Withheld while the manuscript is under review.
- **The datasets.** UCF-11 and UCF-101 are redistributed by their original authors at the links
  above.
- **Trained weights.** Every number in the paper comes from a run of this notebook at one of the
  five fixed seeds, so the notebook plus the seed list is the reproducible artifact.

## Availability

This record is unlisted rather than secret. It is reachable by direct URL for reviewers and
editors, and `robots.txt` together with a `noindex` directive keeps it out of search results
while the manuscript is under review. Both will be removed on acceptance.
