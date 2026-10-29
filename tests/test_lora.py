"""Smoke tests for the Mini-VTAB Query/Value LoRA workflow."""

import torch

from experiments.lora.config import (
    CLASSIFIER_LEARNING_RATE,
    LORA_LEARNING_RATE,
    LORA_RANK,
)
from experiments.lora.lora_layer import LoRALinear
from experiments.lora.model import ViTLoRAClassifier
from experiments.lora.run_lora import build_optimizer, build_scheduler
from vit_core.pretrained import load_pretrained_vit


def test_lora_layer_starts_as_base_layer() -> None:
    layer = LoRALinear(5, 3, rank=2, alpha=2.0)
    inputs = torch.randn(4, 5)
    torch.testing.assert_close(layer(inputs), layer.base(inputs))


def test_lora_model_and_optimizer() -> None:
    num_classes = 10
    model = ViTLoRAClassifier(num_classes=num_classes)
    loaded_tensors = load_pretrained_vit(model)
    assert loaded_tensors == 198
    assert model.head.classifier.out_features == num_classes

    lora_parameters = 12 * 2 * 2 * 768 * LORA_RANK
    classifier_parameters = 768 * num_classes + num_classes
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable_parameters == lora_parameters + classifier_parameters

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(
        ".lora_a." in name
        or ".lora_b." in name
        or name.startswith("head.classifier.")
        for name in trainable_names
    )

    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        LORA_LEARNING_RATE,
        CLASSIFIER_LEARNING_RATE,
    ]
    scheduler = build_scheduler(optimizer, num_epochs=100, warmup_epochs=10)
    assert scheduler is not None


if __name__ == "__main__":
    test_lora_layer_starts_as_base_layer()
    test_lora_model_and_optimizer()
    print("Mini-VTAB LoRA smoke tests passed.")
