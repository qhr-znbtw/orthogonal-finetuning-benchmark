# Mini-VTAB-3: Parameter-Efficient ViT Transfer Learning Benchmark

## Abstract

This report compares five transfer-learning strategies on a controlled three-task image-classification benchmark: Linear Probe, LoRA, block-diagonal OFT, BOFT, and strict GOFT. Every experiment uses the same ImageNet-21k-pretrained ViT-B/16 backbone, the same deterministic 800-example training and 200-example validation protocol, and the same task-specific test sets. Adapter methods modify only the Query and Value projections in the 12 Transformer encoder blocks.

BOFT obtains the highest mean test accuracy at 71.91%, while LoRA follows closely at 71.17% with substantially fewer trainable parameters and shorter training time. Strict GOFT reaches 69.48% mean accuracy with only 18,408 adapter parameters, demonstrating the strongest parameter compression, although its current implementation is slower than the other adapters. These results show that accuracy, parameter efficiency, and computational efficiency are distinct objectives.

## 1. Research questions

The experiments address four questions:

1. How much does adapting the pretrained backbone improve over a frozen Linear Probe?
2. How do additive low-rank updates compare with multiplicative orthogonal updates?
3. How does increasing the expressiveness of an orthogonal parameterization affect accuracy and resource cost?
4. Can an extremely small strict-orthogonal adapter retain most of the benefit of larger methods?

## 2. Experimental protocol

### 2.1 Backbone

- Model: `google/vit-base-patch16-224-in21k`
- Input resolution: 224 x 224 RGB
- Patch size: 16 x 16
- Hidden dimension: 768
- Encoder blocks: 12
- Attention heads: 12
- Pretraining: supervised ImageNet-21k

The project implements the ViT architecture directly in PyTorch and maps the official pretrained tensors into the custom model. The pretrained backbone weights remain frozen for all methods in this report.

### 2.2 Tasks and splits

| Task | Domain | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| DTD | Natural textures | 47 | 800 | 200 | 1,880 |
| EuroSAT | Specialized satellite imagery | 10 | 800 | 200 | 5,400 |
| dSprites Orientation | Structured synthetic reasoning | 16 | 800 | 200 | 73,728 |

The benchmark is a compact, locally reproducible three-task subset inspired by VTAB-1K. It is not the full official 19-task VTAB benchmark. DTD uses its official train, validation, and test partitions; EuroSAT uses a deterministic stratified 60/20/20 source split before the 800/200 subsampling; dSprites uses prepared orientation-class partitions.

### 2.3 Preprocessing

All methods use the same task-independent preprocessing:

1. resize to 224 x 224 with bilinear interpolation;
2. convert to a tensor;
3. normalize every RGB channel with mean 0.5 and standard deviation 0.5.

No random crop, flip, color jitter, or task-specific augmentation is used.

### 2.4 Shared training settings

| Setting | Value |
| --- | --- |
| Epochs | 100 |
| Training batch size | 64 |
| Evaluation batch size | 128 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | 10-epoch linear warmup followed by cosine decay |
| Loss | Cross-entropy |
| Precision | BF16 automatic mixed precision when supported |
| Random seed | 42 |
| Model selection | Highest validation accuracy |
| Hardware | NVIDIA GeForce RTX 4070 Ti SUPER |

The final test result is computed only after restoring the checkpoint with the highest validation accuracy.

## 3. Compared methods

### 3.1 Linear Probe

The complete ViT backbone is frozen and only a new task-specific linear classifier is trained. This provides the baseline for measuring whether a method genuinely adapts the representation.

### 3.2 LoRA

Rank-8 additive low-rank branches are attached to every Query and Value projection:

```text
W' = W + (alpha / r) B A
```

The configuration uses rank `r=8`, `alpha=16`, adapter learning rate `1e-4`, and classifier learning rate `1e-3`.

### 3.3 Block OFT

Each Query and Value projection is preceded by a block-diagonal orthogonal transform. Block size 24 divides the 768-dimensional feature space into 32 independent blocks. Every block is generated from a trainable skew-symmetric matrix through the Cayley transform. The OFT learning rate is `1e-4`; the classifier learning rate is `1e-3`.

### 3.4 BOFT

BOFT composes six block-orthogonal factors with fixed butterfly permutations. The block size remains 24, while the effective information coverage expands across factors. This configuration uses learning rate `1e-4` for BOFT and `1e-3` for the classifier.

### 3.5 Strict GOFT

GOFT replaces dense block rotations with a parallel tree of Givens rotations. A 768-dimensional projection learns exactly `d - 1 = 767` angles evaluated in 10 stages. Across 24 Query/Value projections, the adapter contains only 18,408 trainable parameters. Angles are zero-initialized and parameterized as `theta = pi * tanh(raw_angle)`. No scaling vector and no soft orthogonality regularizer are used, so this is strict GOFT rather than qGOFT. Both GOFT and the classifier use learning rate `1e-3`.

## 4. Results

### 4.1 Test accuracy

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 60.28% |
| LoRA (Q/V, rank 8) | 69.68% | 94.24% | 49.58% | 71.17% |
| Block OFT (Q/V, block 24) | 70.27% | 93.76% | 46.07% | 70.03% |
| BOFT (Q/V, block 24, 6 factors) | **70.32%** | **95.41%** | **49.99%** | **71.91%** |
| Strict GOFT (Q/V) | 69.52% | 93.52% | 45.41% | 69.48% |

### 4.2 Best validation checkpoints

| Method | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Linear Probe | 66.50% at epoch 39 | 92.50% at epoch 69 | 25.50% at epoch 75 |
| LoRA | 69.00% at epoch 34 | 94.50% at epoch 61 | 49.50% at epoch 61 |
| Block OFT | 68.50% at epoch 44 | 93.00% at epoch 69 | 44.00% at epoch 54 |
| BOFT | 67.50% at epoch 18 | 98.00% at epoch 42 | 53.50% at epoch 77 |
| Strict GOFT | 65.00% at epoch 25 | 93.00% at epoch 29 | 45.00% at epoch 50 |

Because the validation set contains only 200 examples, one prediction corresponds to 0.5 percentage points. Small validation differences should therefore not be overinterpreted.

### 4.3 Parameter efficiency

| Method | Adapter parameters | DTD total trainable | EuroSAT total trainable | dSprites total trainable |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 0 | 36,143 | 7,690 | 12,304 |
| LoRA | 294,912 | 331,055 | 302,602 | 307,216 |
| Block OFT | 211,968 | 248,111 | 219,658 | 224,272 |
| BOFT | 1,271,808 | 1,307,951 | 1,279,498 | 1,284,112 |
| Strict GOFT | **18,408** | 54,551 | 26,098 | 30,712 |

The total trainable count includes the task-specific classifier, whose size depends on the number of classes. GOFT uses approximately 16.0 times fewer adapter parameters than LoRA, 11.5 times fewer than block OFT, and 69.1 times fewer than this BOFT configuration.

### 4.4 Training cost

| Method | DTD time | EuroSAT time | dSprites time | Total time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probe | 224.8 s | 174.7 s | 183.0 s | 582.5 s | 1.29 GiB |
| LoRA | 318.1 s | 286.7 s | 284.2 s | 889.0 s | 5.52 GiB |
| Block OFT | 370.1 s | 324.7 s | 320.8 s | 1,015.6 s | 5.52 GiB |
| BOFT | 529.5 s | 481.6 s | 455.1 s | 1,466.2 s | 7.72 GiB |
| Strict GOFT | 750.2 s | 425.4 s | 701.3 s | 1,876.9 s | 5.74 GiB |

GOFT has the fewest adapter parameters but the longest total training time. Its effective frozen weight is reconstructed through multiple sparse rotation stages during every training forward pass. This illustrates that parameter efficiency does not automatically imply runtime efficiency.

## 5. Discussion

### 5.1 Backbone adaptation matters most on structured tasks

On DTD and EuroSAT, Linear Probe is already relatively strong and adaptation provides gains of roughly 2-6 percentage points. On dSprites Orientation, Linear Probe reaches only 24.48%, whereas LoRA and BOFT approach 50%. The structured task therefore provides the clearest evidence that changing internal representations is necessary.

### 5.2 BOFT gives the best accuracy in this configuration

BOFT ranks first on all three test sets and achieves the best mean accuracy. Its advantage over LoRA is small: 0.15 points on DTD, 1.17 points on EuroSAT, and 0.41 points on dSprites. That gain comes with approximately 4.3 times more adapter parameters than LoRA, higher peak memory, and longer training.

### 5.3 LoRA has the strongest practical balance

LoRA obtains 71.17% mean accuracy, only 0.74 points below BOFT, while training faster and using far fewer parameters. Under the current implementation and hardware, LoRA provides the most attractive balance among accuracy, simplicity, memory, and runtime.

### 5.4 Strict orthogonality creates a capacity trade-off

Block OFT and GOFT constrain adaptation to rotations, which preserve norms and pairwise angular geometry. This can help retain pretrained structure but limits the ability to represent large domain shifts. GOFT is especially compressed: despite using only 18,408 adapter parameters, it remains within 0.80 points of BOFT on DTD and gains 20.93 points over Linear Probe on dSprites. Its lower dSprites result relative to LoRA and BOFT suggests that norm adjustment or relaxed orthogonality may be useful. qGOFT is a natural next experiment.

## 6. Limitations

1. Every result uses one random seed, so no confidence interval or standard deviation is available.
2. This is a custom three-task benchmark rather than the complete official VTAB-1K suite.
3. Hyperparameters were fixed from method defaults and prior work; no systematic search was performed.
4. All adapters target Query and Value only. The conclusions do not directly transfer to all-linear-layer configurations.
5. No data augmentation is used, which improves experimental control but may reduce absolute accuracy.
6. Training time reflects the current from-scratch PyTorch implementations and one RTX 4070 Ti SUPER; it is not a hardware-independent property of each method.
7. The 200-example validation set produces coarse 0.5-point accuracy increments and can make checkpoint selection noisy.

## 7. Reproducibility

Download the three datasets and pretrained backbone:

```bash
python -m scripts.download_mini_vtab
python -m scripts.prepare_dsprites_orientation
python -m scripts.download_pretrained_model
```

Run one method on one task:

```bash
python -m experiments.linear_probe.run_linear_probe --task dtd
python -m experiments.lora.run_lora --task dtd
python -m experiments.oft.run_oft --task dtd
python -m experiments.boft.run_boft --task dtd
python -m experiments.goft.run_goft --task dtd
```

Valid task names are `dtd`, `eurosat`, and `dsprites_orientation`. Each JSON file under `results/` stores the full 100-epoch history, selected checkpoint statistics, test metrics, parameter counts, wall-clock measurements, and peak allocated GPU memory.

## 8. Conclusion

The experiment establishes a reproducible comparison across additive and orthogonal parameter-efficient transfer methods. BOFT obtains the highest accuracy, LoRA offers the strongest overall practical trade-off, and GOFT achieves the most aggressive parameter compression. The results support treating accuracy, storage, runtime, and geometry preservation as separate axes rather than searching for one universally best adapter.

## References

1. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
2. Zhai et al. *A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark.* 2019.
3. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
4. Qiu et al. *Controlling Text-to-Image Diffusion by Orthogonal Finetuning.* NeurIPS 2023.
5. Liu et al. *Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization.* ICLR 2024.
6. Ma et al. *Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation.* ICML 2024.
