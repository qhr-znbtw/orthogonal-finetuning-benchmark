"""Tests for strict GOFT and its Mini-VTAB ViT integration."""

import math

import torch
from torch.nn import functional as F

from experiments.goft.config import CLASSIFIER_LEARNING_RATE, GOFT_LEARNING_RATE
from experiments.goft.goft_layer import GOFTLinear, _build_parallel_givens_schedule
from experiments.goft.model import ViTGOFTClassifier
from experiments.goft.run_goft import build_optimizer, build_scheduler, count_parameters
from vit_core.pretrained import load_pretrained_vit


def test_schedule_and_math() -> None:
    left, right, offsets = _build_parallel_givens_schedule(12)
    assert left.numel() == right.numel() == 11
    assert offsets.tolist() == [0, 6, 9, 10, 11]

    layer = GOFTLinear(12, 7)
    inputs = torch.randn(2, 5, 12)
    torch.testing.assert_close(
        layer(inputs),
        F.linear(inputs, layer.base.weight, layer.base.bias),
    )

    with torch.no_grad():
        layer.raw_angles.normal_(mean=0.0, std=0.1)
    rotation = layer.dense_orthogonal_matrix()
    assert torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(12),
        atol=2e-5,
        rtol=2e-5,
    )
    expected = F.linear(layer.rotate_features(inputs), layer.base.weight, layer.base.bias)
    assert torch.allclose(layer(inputs), expected, atol=2e-5, rtol=2e-5)


def test_vit_goft_model_and_optimizer() -> None:
    num_classes = 10
    model = ViTGOFTClassifier(num_classes=num_classes)
    assert load_pretrained_vit(model) == 198

    goft_parameters = 12 * 2 * (768 - 1)
    classifier_parameters = 768 * num_classes + num_classes
    counts = count_parameters(model)
    assert counts["goft"] == goft_parameters == 18_408
    assert counts["classifier"] == classifier_parameters
    assert counts["trainable"] == goft_parameters + classifier_parameters

    first_layer = next(module for module in model.modules() if isinstance(module, GOFTLinear))
    assert first_layer.num_stages == math.ceil(math.log2(768)) == 10

    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        GOFT_LEARNING_RATE,
        CLASSIFIER_LEARNING_RATE,
    ]
    scheduler = build_scheduler(optimizer, num_epochs=100, warmup_epochs=10)
    assert scheduler is not None


if __name__ == "__main__":
    test_schedule_and_math()
    test_vit_goft_model_and_optimizer()
    print("Mini-VTAB GOFT tests passed.")
