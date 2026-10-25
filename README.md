# ViT-LoRA for Oxford-IIIT Pet Classification

A compact, from-scratch implementation of parameter-efficient transfer learning for fine-grained image classification. The project adapts an ImageNet-pretrained ViT-B/16 to the 37-class Oxford-IIIT Pet dataset by training only LoRA branches in the attention layers and a new classification head.

## Results

| Metric | Result |
| --- | ---: |
| Best validation accuracy | **95.52%** |
| Official test accuracy | **93.38%** |
| Correct test predictions | **3,426 / 3,669** |
| Total parameters | 86,122,021 |
| Trainable parameters | 323,365 |
| Trainable fraction | **0.375%** |

The model was trained for 20 epochs on a single NVIDIA GeForce RTX 4070 Ti SUPER. The best validation result was reached at epoch 9. See [research_report.md](research_report.md) for the complete training history, experimental setup, and discussion.

## Method

- **Pretrained backbone:** `google/vit-base-patch16-224`
- **Input resolution:** `224 x 224` RGB
- **Transformer:** 12 encoder blocks, hidden size 768, 12 attention heads
- **LoRA targets:** Query and Value projections in every encoder block
- **LoRA rank:** 8
- **LoRA alpha:** 8
- **Task head:** `LayerNorm + Linear(768, 37)`
- **Optimized parameters:** LoRA branches and the classification head only

Instead of wrapping a complete off-the-shelf classification model, this repository implements the ViT embedding layer, multi-head self-attention, MLP, encoder block, and LoRA linear layer directly in PyTorch. A transparent parameter-name mapping then loads the official Hugging Face ViT-B/16 weights into the custom architecture.

## Repository Structure

```text
.
├── dataset.py                    # Stratified split and data loaders
├── download_pet_dataset.py       # Oxford-IIIT Pet download script
├── download_pretrained_model.py  # ViT-B/16 weight download script
├── load_pretrained_model.py      # Pretrained parameter mapping and loading
├── main.py                       # End-to-end training and evaluation entry point
├── net.py                        # ViT-B/16 and LoRA implementation
├── test.py                       # Validation and test loop
├── train.py                      # Single-epoch and multi-epoch training loops
├── research_report.md            # Full report and epoch-by-epoch results
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
   python download_pet_dataset.py
   ```

2. Download the pretrained ViT-B/16 weights:

   ```bash
   python download_pretrained_model.py
   ```

3. Train the model and evaluate the best checkpoint:

   ```bash
   python main.py
   ```

The best validation checkpoint is saved to `checkpoints/best_model.pt`. The random seed is fixed at 42. The official `trainval` split is divided into stratified training and validation subsets at an 80:20 ratio, while the official test split is reserved for final evaluation.

## Training Configuration

| Hyperparameter | Value |
| --- | ---: |
| Epochs | 20 |
| Training batch size | 32 |
| Validation/test batch size | 64 |
| LoRA learning rate | `1e-4` |
| Classification-head learning rate | `1e-3` |
| Weight decay | `0.01` |
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Random seed | 42 |

Training uses BF16 automatic mixed precision when supported by the selected CUDA device.

## Scope and Limitations

This experiment demonstrates that the custom ViT-LoRA pipeline can perform fine-grained pet classification while updating fewer than 0.4% of the model parameters. It does **not** yet establish that LoRA is superior to other transfer-learning strategies. A frozen-backbone linear-head baseline, full fine-tuning, rank ablations, and repeated runs with multiple random seeds remain to be completed.

## Planned Experiments

- Frozen backbone with a trainable linear classification head
- LoRA rank comparison: `r = 2, 4, 8, 16`
- Query-only, Value-only, and Query-plus-Value LoRA
- Full-parameter fine-tuning
- Mean and standard deviation over multiple random seeds

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
