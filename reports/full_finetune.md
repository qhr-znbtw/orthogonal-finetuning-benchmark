# Full ViT-B/16 Fine-tuning on Oxford-IIIT Pet

## 1. Experiment

This experiment fine-tunes every parameter of the ImageNet-pretrained ViT-B/16 together with a new 37-class classifier. It uses the same dataset split, image preprocessing, random seed, validation rule, 20-epoch budget, and official test set as the parameter-efficient experiments.

The backbone uses a conservative learning rate of `2e-5`, while the randomly initialized classifier uses `1e-3`. A training batch size of 16 is used to keep full backpropagation within the available GPU memory.

## 2. Configuration

| Item | Value |
|---|---:|
| Pretrained model | `google/vit-base-patch16-224` |
| Training samples | 2,944 |
| Validation samples | 736 |
| Test samples | 3,669 |
| Epochs | 20 |
| Training batch size | 16 |
| Evaluation batch size | 64 |
| Optimizer | AdamW |
| Backbone learning rate | `2e-5` |
| Classifier learning rate | `1e-3` |
| Weight decay | `0.01` |
| Random seed | 42 |

## 3. Parameter Cost

| Metric | Value |
|---|---:|
| Total parameters | 85,827,109 |
| Trainable parameters | 85,827,109 |
| Trainable fraction | 100% |
| Checkpoint size | 343,371,590 bytes (about 327.5 MiB) |

Unlike Linear Probe and LoRA, the task checkpoint contains the entire model rather than only a small set of task-specific parameters.

## 4. Results

| Metric | Result |
|---|---:|
| Best validation epoch | 5 |
| Best validation accuracy | **95.38%** |
| Test loss | 0.2453 |
| Test accuracy | **93.24%** |
| Correct test predictions | **3,421 / 3,669** |

## 5. Training History

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.5917 | 85.46% | 0.2074 | 94.16% |
| 2 | 0.0883 | 97.69% | 0.2141 | 93.21% |
| 3 | 0.0323 | 99.29% | 0.1943 | 94.43% |
| 4 | 0.0135 | 99.86% | 0.1778 | 94.29% |
| 5 | 0.0094 | 99.86% | 0.1801 | **95.38%** |
| 6 | 0.0036 | 99.97% | 0.1973 | 94.70% |
| 7 | 0.0078 | 99.93% | 0.1695 | 95.38% |
| 8 | 0.0015 | 100.00% | 0.1643 | 95.24% |
| 9 | 0.0015 | 100.00% | 0.1707 | 95.11% |
| 10 | 0.0207 | 99.35% | 0.2839 | 92.80% |
| 11 | 0.0261 | 99.32% | 0.2527 | 93.07% |
| 12 | 0.0268 | 99.15% | 0.2494 | 93.61% |
| 13 | 0.0422 | 98.88% | 0.3742 | 92.93% |
| 14 | 0.0562 | 98.47% | 0.4476 | 91.58% |
| 15 | 0.0197 | 99.69% | 0.2828 | 94.02% |
| 16 | 0.0063 | 99.90% | 0.3485 | 92.39% |
| 17 | 0.0094 | 99.83% | 0.3753 | 92.26% |
| 18 | 0.0092 | 99.80% | 0.2982 | 93.89% |
| 19 | 0.0607 | 99.05% | 0.3591 | 91.98% |
| 20 | 0.0095 | 99.83% | 0.3639 | 92.80% |

## 6. Interpretation

Full fine-tuning improves test accuracy over Linear Probe by 0.63 percentage points and produces 23 additional correct predictions. It nevertheless finishes 0.14 percentage points below the current LoRA result while updating roughly 265 times as many parameters as LoRA.

The optimization becomes unstable after epoch 9: training accuracy is already near 100%, while validation loss and accuracy fluctuate sharply. The restored epoch-5 checkpoint prevents this late degradation from affecting the final test result. This suggests that full fine-tuning would benefit more than the parameter-efficient methods from a learning-rate schedule, warmup, stronger regularization, or earlier stopping.

The comparison is still a single-seed observation. In addition, Full Fine-tuning uses batch size 16 while Linear Probe and LoRA use batch size 32, so it should not be interpreted as a perfectly controlled optimizer ablation.

## 7. Reproduction

```powershell
cd D:\ViT-Transfer-Learning
E:\anaconda\python.exe -m experiments.full_finetune.run_full_finetune
```

The machine-readable result is saved in `results/full_finetune.json`, and the best complete-model checkpoint is saved in `checkpoints/full_finetune/best_model.pt`.
