# ViT-B/16 Linear Probe on Oxford-IIIT Pet

## 1. Experiment

This experiment measures how well the frozen ImageNet-pretrained ViT-B/16 representation transfers to Oxford-IIIT Pet. The complete pretrained network is frozen, including the final LayerNorm, and only a newly initialized `Linear(768, 37)` classifier is optimized.

The data split, preprocessing, random seed, optimizer family, number of epochs, batch sizes, validation procedure, and official test set are identical to the LoRA experiment.

## 2. Configuration

| Item | Value |
|---|---:|
| Pretrained model | `google/vit-base-patch16-224` |
| Training samples | 2,944 |
| Validation samples | 736 |
| Test samples | 3,669 |
| Epochs | 20 |
| Training batch size | 32 |
| Evaluation batch size | 64 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Random seed | 42 |

## 3. Parameter Efficiency

| Metric | Value |
|---|---:|
| Total parameters | 85,827,109 |
| Trainable parameters | 28,453 |
| Trainable fraction | 0.0332% |
| Checkpoint size | 115,434 bytes |

The only trainable tensors are `head.classifier.weight` and `head.classifier.bias`.

## 4. Results

| Metric | Result |
|---|---:|
| Best validation epoch | 16 |
| Best validation accuracy | **94.70%** |
| Test loss | 0.2415 |
| Test accuracy | **92.61%** |
| Correct test predictions | **3,398 / 3,669** |

## 5. Training History

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.8751 | 80.67% | 0.2965 | 92.39% |
| 2 | 0.2085 | 95.41% | 0.2311 | 93.34% |
| 3 | 0.1381 | 96.77% | 0.2120 | 94.02% |
| 4 | 0.0978 | 97.83% | 0.1883 | 93.89% |
| 5 | 0.0739 | 98.47% | 0.1871 | 94.29% |
| 6 | 0.0578 | 98.91% | 0.1889 | 94.16% |
| 7 | 0.0501 | 99.32% | 0.1851 | 93.89% |
| 8 | 0.0402 | 99.52% | 0.1727 | 94.16% |
| 9 | 0.0359 | 99.35% | 0.1767 | 93.61% |
| 10 | 0.0332 | 99.59% | 0.1752 | 94.02% |
| 11 | 0.0253 | 99.76% | 0.1735 | 94.29% |
| 12 | 0.0241 | 99.76% | 0.1791 | 94.57% |
| 13 | 0.0221 | 99.80% | 0.1775 | 94.43% |
| 14 | 0.0191 | 99.86% | 0.1754 | 94.16% |
| 15 | 0.0172 | 99.86% | 0.1714 | 94.57% |
| 16 | 0.0163 | 99.90% | 0.1747 | **94.70%** |
| 17 | 0.0137 | 99.93% | 0.1717 | 94.29% |
| 18 | 0.0128 | 99.86% | 0.1758 | 94.29% |
| 19 | 0.0133 | 99.93% | 0.1777 | 94.16% |
| 20 | 0.0099 | 99.97% | 0.1765 | 94.29% |

## 6. Interpretation

The frozen ViT representation is already highly transferable: training only 28,453 parameters reaches 92.61% test accuracy. Training accuracy approaches 100%, while validation accuracy stabilizes around 94–95%, so later epochs mainly fit the augmented training samples rather than consistently improving generalization.

Compared with the current LoRA run, Linear Probe is 0.77 percentage points lower on the test set but uses about 11.4 times fewer trainable parameters. This establishes a strong baseline: LoRA provides a measurable but modest gain in this single run, rather than transforming an otherwise weak representation.

The comparison is not yet a statistical conclusion because both methods currently use one random seed.

## 7. Reproduction

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m experiments.linear_probe.run_linear_probe
```

The machine-readable result is saved in `results/linear_probe.json`, and the best classifier checkpoint is saved in `checkpoints/linear_probe/best_model.pt`.
