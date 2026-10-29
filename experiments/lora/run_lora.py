"""Train and evaluate Query/Value LoRA on Mini-VTAB-3."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiments.lora.config import (
    CLASSIFIER_LEARNING_RATE,
    LORA_ALPHA,
    LORA_LEARNING_RATE,
    LORA_RANK,
    NUM_EPOCHS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from experiments.lora.model import ViTLoRAClassifier
from vit_core.evaluation import test
from vit_core.mini_vtab import (
    EVAL_BATCH_SIZE,
    RANDOM_SEED,
    TASK_SPECS,
    TRAIN_BATCH_SIZE,
    build_dataloaders,
    get_task_spec,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """Use a conservative LR for adapters and a larger LR for the new head."""
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
            raise RuntimeError(f"Unclassified trainable parameter: {name}")

    if not lora_parameters or not classifier_parameters:
        raise RuntimeError("LoRA and classifier parameter groups must both be non-empty.")

    return AdamW(
        [
            {"params": lora_parameters, "lr": LORA_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    num_epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    """Apply the same warmup/cosine multiplier to both parameter groups."""
    if not 0 <= warmup_epochs < num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs).")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = num_epochs - warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(decay_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    task = get_task_spec(args.task)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "lora"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.pt"
    )
    result_path = (
        PROJECT_ROOT
        / "results"
        / "lora"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.json"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        task.name,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    model = ViTLoRAClassifier(num_classes=task.num_classes)
    loaded_tensors = load_pretrained_vit(model)
    model.set_trainable_parameters()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Task: {task.name} ({task.group}, {task.num_classes} classes)")
    print(f"Device: {device}")
    print(f"Dataset sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"LoRA: Query+Value, rank={LORA_RANK}, alpha={LORA_ALPHA:g}")
    print(f"Schedule: {args.warmup_epochs} warmup epochs + cosine decay")
    print(f"Learning rates: LoRA={LORA_LEARNING_RATE:g}, classifier={CLASSIFIER_LEARNING_RATE:g}")
    print(f"Loaded backbone tensors: {loaded_tensors}")
    print(f"Parameters: {trainable_parameters:,} trainable / {total_parameters:,} total")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_start = time.perf_counter()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    test_start = time.perf_counter()
    test_metrics = test(model, test_loader, criterion, device)
    synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))
    result = {
        "protocol": "development_800_200",
        "method": "lora",
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "lora_targets": ["query", "value"],
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "best_epoch": int(best_epoch["epoch"]),
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "test": test_metrics,
        "history": history,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\nFinal test result")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Peak VRAM: {peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"Result file: {result_path}")


if __name__ == "__main__":
    main()
