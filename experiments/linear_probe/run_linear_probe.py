
"""Train and evaluate the frozen-backbone ViT linear-probe baseline."""

import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from experiments.linear_probe.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from experiments.linear_probe.model import ViTLinearProbe
from vit_core.evaluation import test
from vit_core.pet_dataset import RANDOM_SEED, build_dataloaders
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "linear_probe" / "best_model.pt"
RESULT_PATH = PROJECT_ROOT / "results" / "linear_probe.json"


def set_random_seed(seed: int) -> None:
    """Fix Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """Optimize exactly the linear classification head."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def save_results(
    history: list[dict[str, float | int]],
    test_metrics: dict[str, float | int],
    total_parameters: int,
    trainable_parameters: int,
) -> None:
    """Save reproducible metrics for the later method comparison."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))
    result = {
        "method": "linear_probe",
        "seed": RANDOM_SEED,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "best_epoch": int(best_epoch["epoch"]),
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "test": test_metrics,
        "history": history,
    }
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = ViTLinearProbe()
    load_pretrained_vit(model)
    model.set_trainable_parameters()
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Total parameters: {total_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")

    history = train(
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
    save_results(
        history=history,
        test_metrics=test_metrics,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )

    print("\nFinal test result")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Correct: {test_metrics['correct']} / {test_metrics['total']}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Result file: {RESULT_PATH}")


if __name__ == "__main__":
    main()
