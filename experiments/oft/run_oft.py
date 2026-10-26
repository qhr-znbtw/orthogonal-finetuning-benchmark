
"""Train and evaluate block-diagonal OFT on Oxford-IIIT Pet."""

import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from experiments.oft.config import (
    CLASSIFIER_LEARNING_RATE,
    NUM_EPOCHS,
    OFT_BLOCK_SIZE,
    OFT_LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)
from experiments.oft.model import ViTOFTForPetClassification
from experiments.oft.oft_layer import BlockOFTLinear
from vit_core.evaluation import test
from vit_core.pet_dataset import RANDOM_SEED, build_dataloaders
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "oft" / "best_model.pt"
RESULT_PATH = PROJECT_ROOT / "results" / "oft.json"


def set_random_seed(seed: int) -> None:
    """Fix Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """Use separate learning rates for OFT and the new classifier."""
    oft_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".skew_parameters"):
            oft_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"发现未分类的可训练参数：{name}")

    return AdamW(
        [
            {"params": oft_parameters, "lr": OFT_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, OFT, classifier, and all trainable parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    oft = sum(
        module.skew_parameters.numel()
        for module in model.modules()
        if isinstance(module, BlockOFTLinear)
    )
    classifier = sum(
        parameter.numel() for parameter in model.head.classifier.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != oft + classifier:
        raise RuntimeError("可训练参数计数与OFT加分类头不一致。")
    return {
        "total": total,
        "trainable": trainable,
        "oft": oft,
        "classifier": classifier,
    }


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_results(
    history: list[dict[str, float | int]],
    test_metrics: dict[str, float | int],
    parameter_counts: dict[str, int],
    training_seconds: float,
    test_seconds: float,
) -> None:
    """Persist accuracy, parameter counts, and measured wall-clock times."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))
    result = {
        "method": "block_oft",
        "targets": ["query", "value"],
        "seed": RANDOM_SEED,
        "epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "block_size": OFT_BLOCK_SIZE,
        "num_butterfly_factors": 1,
        "oft_learning_rate": OFT_LEARNING_RATE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "total_parameters": parameter_counts["total"],
        "trainable_parameters": parameter_counts["trainable"],
        "oft_parameters": parameter_counts["oft"],
        "classifier_parameters": parameter_counts["classifier"],
        "trainable_fraction": (
            parameter_counts["trainable"] / parameter_counts["total"]
        ),
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
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

    train_loader, val_loader, test_loader = build_dataloaders(
        train_batch_size=TRAIN_BATCH_SIZE,
    )
    model = ViTOFTForPetClassification()
    load_pretrained_vit(model)
    model.set_trainable_parameters()
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    parameter_counts = count_parameters(model)

    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"OFT block size: {OFT_BLOCK_SIZE}")
    print("Butterfly factors: 1 (block-diagonal OFT)")
    print(f"OFT learning rate: {OFT_LEARNING_RATE}")
    print(f"Classifier learning rate: {CLASSIFIER_LEARNING_RATE}")
    print(f"Total parameters: {parameter_counts['total']}")
    print(f"Trainable parameters: {parameter_counts['trainable']}")
    print(f"OFT parameters: {parameter_counts['oft']}")
    print(f"Classifier parameters: {parameter_counts['classifier']}")

    _synchronize_cuda(device)
    training_start = time.perf_counter()
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
    _synchronize_cuda(device)
    training_seconds = time.perf_counter() - training_start

    _synchronize_cuda(device)
    test_start = time.perf_counter()
    test_metrics = test(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    _synchronize_cuda(device)
    test_seconds = time.perf_counter() - test_start

    save_results(
        history=history,
        test_metrics=test_metrics,
        parameter_counts=parameter_counts,
        training_seconds=training_seconds,
        test_seconds=test_seconds,
    )

    print("\nFinal test result")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Correct: {test_metrics['correct']} / {test_metrics['total']}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Test time: {test_seconds:.2f} seconds")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Result file: {RESULT_PATH}")


if __name__ == "__main__":
    main()
