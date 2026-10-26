# Transfer-Learning Comparison

All completed methods use the same ViT-B/16 pretrained weights, Oxford-IIIT Pet split, preprocessing, random seed 42, 20-epoch budget, and official test set.

| Method | Best validation accuracy | Test accuracy | Correct / total | Trainable parameters | Trainable fraction |
|---|---:|---:|---:|---:|---:|
| Linear Probe | 94.70% | 92.61% | 3,398 / 3,669 | 28,453 | 0.0332% |
| LoRA (Q+V, rank 8) | **95.52%** | 93.38% | 3,426 / 3,669 | 323,365 | 0.3755% |
| BOFT (Q+V, b=24, m=6) | 94.84% | **93.51%** | **3,431 / 3,669** | 1,300,261 | 1.4929% |
| Full Fine-tuning | 95.38% | 93.24% | 3,421 / 3,669 | 85,827,109 | 100% |
| Block OFT (Q+V, b=24) | 95.38% | 93.13% | 3,417 / 3,669 | 240,421 | 0.2794% |

Current observed LoRA gain over Linear Probe:

- Validation accuracy: +0.82 percentage points.
- Test accuracy: +0.77 percentage points.
- Correct test predictions: +28.
- Trainable parameters: about 11.4 times as many as Linear Probe.

These are single-seed observations and should not yet be interpreted as a statistically stable advantage.

Full Fine-tuning gains 0.63 percentage points over Linear Probe but remains 0.14 points below LoRA in the current runs. It updates roughly 265 times as many parameters as LoRA and produces a roughly 327.5 MiB task checkpoint. Full Fine-tuning uses batch size 16, whereas Linear Probe and LoRA use batch size 32.

BOFT reaches the highest current test accuracy by 0.13 percentage points over LoRA, corresponding to five additional correct test predictions. However, BOFT uses about 4.02 times as many trainable parameters as rank-8 LoRA, so this is not a parameter-matched comparison. Its measured 20-epoch training time is 414.02 seconds and final test time is 15.09 seconds on the RTX 4070 Ti SUPER.

Block OFT improves on Linear Probe by 0.52 test percentage points, corresponding to 19 additional correct predictions. It trails LoRA by 0.25 points and BOFT by 0.38 points, while using only 240,421 trainable parameters: 25.65% fewer than rank-8 LoRA and 81.51% fewer than BOFT. Its measured 20-epoch training time is 217.27 seconds, about 52.48% of BOFT's time. OFT versus BOFT isolates the effect of adding five more permuted orthogonal factors, but it is a capacity comparison rather than a parameter-matched ablation.
