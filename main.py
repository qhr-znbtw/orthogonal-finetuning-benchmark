"""Oxford-IIIT Pet ViT-LoRA实验的运行入口。"""

import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from dataset import RANDOM_SEED, build_dataloaders
from load_pretrained_model import load_pretrained_model
from test import test
from train import train


PROJECT_DIR = Path(__file__).parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_model.pt"

NUM_EPOCHS = 20
LORA_LEARNING_RATE = 1e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01


def set_random_seed(seed: int) -> None:
    """固定Python和PyTorch的随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """为LoRA和随机初始化的分类头设置不同学习率。"""
    lora_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if ".lora_a." in name or ".lora_b." in name:
            lora_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"发现未分类的可训练参数：{name}")

    return AdamW(
        [
            {"params": lora_parameters, "lr": LORA_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = load_pretrained_model()
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"LoRA learning rate: {LORA_LEARNING_RATE}")
    print(f"Classifier learning rate: {CLASSIFIER_LEARNING_RATE}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )

    test_metrics = test(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test result")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Correct: {test_metrics['correct']} / {test_metrics['total']}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
