# ViT Transfer Learning for Oxford-IIIT Pet Classification

A compact, from-scratch comparison of transfer-learning strategies for fine-grained image classification. The project adapts an ImageNet-pretrained ViT-B/16 to the 37-class Oxford-IIIT Pet dataset using Linear Probe, LoRA, block OFT, BOFT, and Full Fine-tuning.

## Results

| Method | Best validation accuracy | Official test accuracy | Trainable parameters | Trainable fraction |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 94.70% | 92.61% | 28,453 | 0.0332% |
| LoRA (Q+V, rank 8) | **95.52%** | 93.38% | 323,365 | 0.3755% |
| BOFT (Q+V, b=24, m=6) | 94.84% | **93.51%** | 1,300,261 | 1.4929% |
| Block OFT (Q+V, b=24) | 95.38% | 93.13% | 240,421 | 0.2794% |
| Full Fine-tuning | 95.38% | 93.24% | 85,827,109 | 100% |

All completed methods were trained for 20 epochs on a single NVIDIA GeForce RTX 4070 Ti SUPER. See the [comparison report](reports/comparison.md) and individual reports in `reports/` for full histories and discussion.

The complete motivation, implementation, results, limitations, and proposed ablations are documented in the [Chinese project report](reports/project_report.md).

## Implemented Methods

- **Pretrained backbone:** `google/vit-base-patch16-224`
- **Input resolution:** `224 x 224` RGB
- **Transformer:** 12 encoder blocks, hidden size 768, 12 attention heads
- **Task head:** `LayerNorm + Linear(768, 37)`
- **Linear Probe:** train only the new classifier
- **LoRA:** train rank-8 Query/Value adapters and the classifier
- **Block OFT:** train one block-size-24 Query/Value orthogonal factor and the classifier
- **BOFT:** train six-factor, block-size-24 Query/Value orthogonal butterfly adapters and the classifier
- **Full Fine-tuning:** train the complete backbone and classifier

Instead of wrapping a complete off-the-shelf classification model, this repository implements the ViT embedding layer, multi-head self-attention, MLP, encoder block, and LoRA linear layer directly in PyTorch. A transparent parameter-name mapping then loads the official Hugging Face ViT-B/16 weights into the custom architecture.

## Repository Structure

```text
.
├── experiments/
│   ├── lora/                     # Working ViT-LoRA experiment
│   ├── boft/                     # Orthogonal butterfly Q/V adaptation
│   ├── linear_probe/             # Frozen-backbone baseline
│   ├── oft/                      # Block-diagonal OFT experiment
│   └── full_finetune/            # Full fine-tuning experiment
├── vit_core/                     # Shared data, training, and evaluation code
├── scripts/                      # Dataset and weight download commands
├── reports/                      # Per-method and comparison reports
├── tests/test_lora.py            # End-to-end LoRA smoke test
├── tests/test_boft.py            # BOFT algebra and integration tests
├── tests/test_oft.py             # OFT algebra and shared-solve tests
├── checkpoints/                  # Generated experiment checkpoints
└── requirements.txt
```

Datasets, pretrained weights, caches, and generated checkpoints are intentionally excluded from Git. They can be downloaded or reproduced locally with the provided scripts.

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build compatible with your CUDA environment from the [official PyTorch installation page](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Experiment

1. Download Oxford-IIIT Pet:

   ```bash
   python -m scripts.download_pet_dataset
   ```

2. Download the pretrained ViT-B/16 weights:

   ```bash
   python -m scripts.download_pretrained_model
   ```

3. Train the model and evaluate the best checkpoint:

   ```bash
   python -m experiments.lora.run_lora
   ```

The best validation checkpoint is saved to `checkpoints/lora/best_model.pt`. The random seed is fixed at 42. The official `trainval` split is divided into stratified training and validation subsets at an 80:20 ratio, while the official test split is reserved for final evaluation.

To verify the complete local workflow without starting another 20-epoch run:

```bash
python -m tests.test_lora
```

Run or verify BOFT with:

```bash
python -m tests.test_boft
python -m experiments.boft.run_boft
```

Run or verify block OFT with:

```bash
python -m tests.test_oft
python -m experiments.oft.run_oft
```

## Training Configuration

| Hyperparameter | Value |
| --- | ---: |
| Epochs | 20 |
| Training batch size | 32 |
| Validation/test batch size | 64 |
| LoRA learning rate | `1e-4` |
| OFT learning rate | `1e-4` |
| OFT block size / factors | `24 / 1` |
| BOFT learning rate | `1e-4` |
| BOFT block size / factors | `24 / 6` |
| Classification-head learning rate | `1e-3` |
| Weight decay | `0.01` |
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Random seed | 42 |

Training uses BF16 automatic mixed precision when supported by the selected CUDA device.

## Scope and Limitations

The completed experiments show that all five methods transfer strongly. In this single-seed comparison, BOFT has the highest test accuracy at 93.51%, 0.13 percentage points above LoRA, but uses about 4.02 times as many trainable parameters. Block OFT reaches 93.13% with only 240,421 trainable parameters and roughly half BOFT's measured training time. Parameter-matched ablations and repeated runs remain necessary for a broader conclusion.

## Planned Experiments

- LoRA rank comparison: `r = 2, 4, 8, 16`
- Query-only, Value-only, and Query-plus-Value LoRA
- Parameter-matched BOFT versus LoRA
- Mean and standard deviation over multiple random seeds

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
