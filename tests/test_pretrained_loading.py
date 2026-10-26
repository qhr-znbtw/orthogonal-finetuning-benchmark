
"""Tests for the shared ViT backbone, pretrained loader, and checkpoints."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from vit_core.checkpoint import (
    load_trainable_checkpoint,
    save_trainable_checkpoint,
)
from vit_core.pretrained import build_vit_weight_map, load_pretrained_vit
from vit_core.vit_backbone import ViTConfig, ViTForImageClassification


def build_small_model() -> ViTForImageClassification:
    """Return a small ViT that keeps unit tests fast."""
    config = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=64,
        num_heads=8,
        intermediate_size=128,
        num_layers=2,
    )
    return ViTForImageClassification(num_classes=5, config=config)


def test_small_vit_forward() -> None:
    model = build_small_model()
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 5)
    assert model.forward_features(images).shape == (2, 17, 64)


def test_checkpoint_round_trip() -> None:
    model = build_small_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.classifier.parameters():
        parameter.requires_grad = True

    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    with TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "model.pt"
        save_trainable_checkpoint(
            model,
            checkpoint_path,
            epoch=3,
            val_accuracy=0.75,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(1.0)

        checkpoint = load_trainable_checkpoint(model, checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["val_accuracy"] == 0.75
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, expected[name])


def test_vit_base_pretrained_loading() -> None:
    model = ViTForImageClassification(num_classes=37)
    weight_map = build_vit_weight_map(model)
    assert len(weight_map) == 198
    assert load_pretrained_vit(model) == 198


if __name__ == "__main__":
    test_small_vit_forward()
    test_checkpoint_round_trip()
    test_vit_base_pretrained_loading()
    print("Shared ViT core tests passed.")
