# Strict GOFT on Mini-VTAB-3

## Objective

This experiment adds strict Givens Orthogonal Fine-Tuning (GOFT) to the existing ViT-B/16 transfer-learning benchmark. The goal is to measure how much downstream adaptation can be obtained from a genuinely orthogonal adapter with only O(d) trainable parameters per target matrix.

For a 768-dimensional projection, GOFT learns 767 angles. Each angle defines a two-dimensional Givens rotation. The rotations are arranged as a balanced tree: adjacent coordinates are paired first, then the representatives of increasingly distant groups are paired. Since 768 is not a power of two, an unpaired final group is carried into the next stage. The resulting schedule contains exactly 767 rotations in 10 parallel stages.

The learned angles are parameterized as

```text
theta = pi * tanh(raw_angle)
```

and every `raw_angle` is initialized to zero. Therefore, each adapter is exactly the identity at initialization. No learned scaling and no soft orthogonality regularizer are used, so this experiment is strict GOFT rather than qGOFT.

## Fair-comparison configuration

The ICML 2024 paper applies GOFT to all linear layers in its VTAB setup. This project instead targets Query and Value only, matching the existing LoRA, block OFT, and BOFT experiments. Consequently, these numbers measure a controlled Q/V-only comparison and should not be presented as a reproduction of the paper's full VTAB table.

| Setting | Value |
| --- | --- |
| Backbone | ViT-B/16, ImageNet-21k pretrained |
| Target modules | Query and Value in all 12 encoder blocks |
| GOFT parameters per projection | 767 |
| Parallel stages | 10 |
| Total GOFT parameters | 18,408 |
| GOFT learning rate | 1e-3 |
| Classifier learning rate | 1e-3 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Schedule | 10-epoch linear warmup, then cosine decay |
| Epochs | 100 |
| Train/eval batch size | 64 / 128 |
| Random seed | 42 |
| Precision | BF16 automatic mixed precision |

## Results

All test metrics use the checkpoint with the highest validation accuracy.

| Task | Best epoch | Best validation accuracy | Test accuracy | Total trainable parameters | Training time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 25 | 65.00% | 69.52% | 54,551 | 750.2 s | 5.74 GiB |
| EuroSAT | 29 | 93.00% | 93.52% | 26,098 | 425.4 s | 5.74 GiB |
| dSprites Orientation | 50 | 45.00% | 45.41% | 30,712 | 701.3 s | 5.74 GiB |

The trainable-parameter totals include the task-specific classifier. The adapter itself always contains 18,408 parameters.

## Comparison

| Method | DTD | EuroSAT | dSprites Orientation | Adapter parameters |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 0 |
| LoRA | 69.68% | 94.24% | 49.58% | 294,912 |
| Block OFT | 70.27% | 93.76% | 46.07% | 211,968 |
| BOFT | 70.32% | 95.41% | 49.99% | 1,271,808 |
| GOFT | 69.52% | 93.52% | 45.41% | 18,408 |

GOFT uses 16 times fewer adapter parameters than rank-8 Q/V LoRA, approximately 11.5 times fewer than block OFT, and approximately 69.1 times fewer than the six-factor BOFT configuration. Despite this compression, it remains within 0.80 percentage points of the best method on DTD and improves dSprites Orientation by 20.93 points over Linear Probe.

Strict GOFT is not the accuracy winner in this single-seed benchmark. Its result is instead a parameter-efficiency result: a very small, exactly orthogonal adapter retains most of the benefit of substantially larger methods. The remaining gap on dSprites is consistent with the restriction that a pure rotation cannot change vector norms. qGOFT is the natural next experiment because it relaxes that restriction under a soft orthogonality penalty.

GOFT is also not the fastest method here. The adapter has few parameters, but the frozen effective weight must be reconstructed through multiple sparse rotation stages during every training forward pass. Parameter efficiency and computational efficiency are therefore different quantities.

## Verification

`tests/test_goft.py` verifies:

1. a dimension-d schedule contains exactly d - 1 pairs;
2. zero initialization reproduces the frozen base layer exactly;
3. the materialized rotation satisfies `R.T @ R = I`;
4. the sparse effective-weight implementation equals explicit feature rotation;
5. ViT-B/16 contains 24 Q/V GOFT projections and exactly 18,408 GOFT parameters;
6. only GOFT angles and the classifier are trainable.

Run the verification with:

```bash
python -m tests.test_goft
```

## Reference

Xinyu Ma, Xu Chu, Zhibang Yang, Yang Lin, Xin Gao, and Junfeng Zhao. "Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation." ICML 2024. <https://proceedings.mlr.press/v235/ma24a.html>
