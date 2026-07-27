# SAM-CapsNet

**Capsule Network with Multi-Scale Attention for Video-Based Human Activity Recognition**

This repository is the replication record for the manuscript. It contains the single
self-contained notebook that defines the model, the ablation grid, the baselines and the
analysis code that produces every table and figure in the paper.

> **The manuscript is under review, so no accuracy values are published here.** The notebook is
> distributed with its outputs cleared. Running it end to end regenerates the tables and figures
> from your own copy of UCF-101.

[`Capsule_Network_with_Multi_Scale_Attention_(SAM_CapsNet)_for_Video_Based_Human_Activity_Recognition.ipynb`](Capsule_Network_with_Multi_Scale_Attention_(SAM_CapsNet)_for_Video_Based_Human_Activity_Recognition.ipynb)

A reader-friendly summary of the architecture, the run configurations and the cell-to-artifact
map is at **https://ycwonglab.github.io/SAM_CapsNet/**.

## Architecture

A shared four-layer convolutional stem encodes every frame independently to a 256-channel
feature map. Each frame is then refined by the Multi-Scale SAM block, and the refined frames are
mean-pooled over time before the capsule stage.

The attention block applies four steps in a fixed order:

1. **Channel.** CBAM-style squeeze-excitation. Average and max pooling share one bottleneck MLP
   (reduction ratio 8), summed and passed through a sigmoid, applied multiplicatively.
2. **Temporal motion map.** Frame `t` is concatenated with frame `t-1` and passed through two
   parallel convolutions at different receptive fields (3x3 and 5x5), fused by a 1x1 convolution
   and a sigmoid. Frame 0 has no predecessor and so receives no motion map.
3. **Spatial.** A learned point prompt (initialised at sigma = 0.1) is added to the
   channel-refined features, which are then concatenated with the motion map and passed through
   a 1x1 / 3x3 / 1x1 convolution stack ending in a sigmoid, applied multiplicatively.
4. **Temporal residual.** `out + tanh(alpha) * (out * a_m)`. The `tanh` gate is signed, so motion
   regions can be amplified rather than only suppressed.

The capsule stage is a standard two-layer arrangement. `PrimaryCaps` is a 3x3 convolution
reshaped into 8-dimensional capsules and squashed. `ActivityCaps` runs three dynamic-routing
iterations to one 16-dimensional capsule per class. The class decision is the L2 norm of the
output capsules, trained with cross-entropy on those norms.

## The two run configurations

The capsule transformation tensor `W` is shaped `(in_caps, num_classes, 16, 8)`, so its size
scales with the product of the input-capsule count and the class count. The 101-class run
therefore uses a lower input resolution to stay inside 8 GB of GPU memory.

| | 11-class (Tables 1 and 2) | 101-class (Table 3 row) |
|---|---|---|
| Notebook cell | 7A | 7B |
| Classes | 11 (UCF-101 subset) | 101 (all) |
| Input resolution | 224 x 224 | 112 x 112 |
| Feature map | 14 x 14 | 7 x 7 |
| Primary capsules per cell | 32 | 16 |
| Input capsules to routing | 6272 | 784 |
| Clips | 938 | ~13,320 |

Both configurations use 16 frames per clip sampled at a fixed 5 fps, 15 epochs, batch size 2 with
gradient accumulation 4 (effective batch 8), Adam at learning rate 5e-4 with weight decay 1e-4,
step decay of 0.5 every 7 epochs, gradient clipping at 5.0, mixed precision, and seeds
42 / 123 / 456 / 789 / 2024. The checkpoint with the best validation accuracy is the one
evaluated on test.

The 11-class subset maps to these UCF-101 folders: `Basketball`, `Biking`, `Diving`,
`GolfSwing`, `HorseRiding`, `SoccerJuggling`, `Swing`, `TennisSwing`, `TrampolineJumping`,
`VolleyballSpiking`, `WalkingWithDog` (the paper's "Walking").

## Ablation grid

Eight capsule configurations, registered in Cell 5 and run by Cell 10.

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

Note that `full_sam` is the only configuration carrying the point prompt and the learnable
fusion parameter, so the step from `tm_sp` or `ch_sp` to the proposed model moves more than one
component at a time.

Four non-capsule baselines share the protocol: `BaselineCNN` (identical stem, mean-pooled linear
softmax), `CNNSAMNoCapsule` (stem plus attention, no capsules), `ResNet18Baseline` (randomly
initialised, matching the no-pretrained-backbone protocol) and MobileNetV2.

## Which cell produces which result

| Artifact | Cells |
|---|---|
| Table 1 (full comparison) | 10 and 10B produce the runs, 10C assembles |
| Table 2 (per-class accuracy) | 11 (text table, heatmap, grouped bars) |
| Table 3 (cross-model, with parameter counts) | 8B, 9B and 10 produce the runs, 10D assembles |
| Parameter counts | 8C |
| Frame-count grid search over T | 8D |
| Inference latency | 9C |
| Sec. II-E capsule magnitude and activation attenuation | 14 |
| Fig. 2 coupling-coefficient clustering | 15 |
| Fig. 3 four-panel interpretability (attention, PrimaryCaps, PCA separability, polar) | 16 |
| Fig. 4 inter-class capsule vector distributions | 17 |
| Fig. 5 confusion-matrix comparison | 18 |
| Subject-independent (leave-groups-out) split | 10A |

## Running it

Requires a CUDA GPU with at least 8 GB, PyTorch, and a local copy of UCF-101.

1. Open the notebook and run Cells 1 to 6B to define the dataset, model, ablation registry and
   training utilities.
2. Set `DATASET_PATH` in **Cell 7A** to your UCF-101 root. Cell 8A is a path diagnostic that
   reports what it can and cannot find before you commit to a long run.
3. Run Cell 8 to build the frame cache. The first pass decodes video and is slow. Later passes
   load from `.cache_ucf101_11class/`.
4. Run Cell 9 as a single-seed smoke test before starting the full grid.
5. Run Cell 10, then Cell 10B, then Cell 10C for Table 1.
6. For the 101-class row, switch to **Cell 7B**, re-run Cell 8 with `CLASS_SUBSET = None`, then
   run Cell 9B.

Results are written to `SAVE_DIR` as JSON after every configuration, so an interrupted run can
be resumed and replotted with Cell 12 without retraining.

### Datasets

- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php
- UCF-11 (YouTube Action): https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php

The notebook builds its 11-class set from the UCF-101 folders listed above rather than from the
separate UCF-11 distribution, so one download covers both experiments. The subject-independent
splitter in Cell 10A reads group identifiers from either the filename (`v_Basketball_g01_c01`)
or the parent folder (`Basketball/group01/`), so it works with either layout.

## Deliberately not here

- **Accuracy values, logs and checkpoints.** Withheld while the manuscript is under review.
- **The dataset.** UCF-101 is redistributed by its original authors at the link above.
- **Trained weights.** Every number in the paper comes from a run of this notebook at one of the
  five fixed seeds, so the notebook plus the seed list is the reproducible artifact.

## Availability

This record is unlisted rather than secret. It is reachable by direct URL for reviewers and
editors, and `robots.txt` together with a `noindex` directive keeps it out of search results
while the manuscript is under review. Both will be removed on acceptance.
