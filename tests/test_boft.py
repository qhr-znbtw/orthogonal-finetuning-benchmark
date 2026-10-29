"""Tests for BOFT and its Mini-VTAB ViT integration."""

import torch
from torch.nn import functional as F

from experiments.boft.boft_layer import BOFTLinear
from experiments.boft.config import (
    BOFT_BLOCK_SIZE,
    BOFT_LEARNING_RATE,
    BOFT_NUM_FACTORS,
    CLASSIFIER_LEARNING_RATE,
)
from experiments.boft.model import ViTBOFTClassifier
from experiments.boft.run_boft import build_optimizer, build_scheduler, count_parameters
from vit_core.pretrained import load_pretrained_vit


def test_boft_math() -> None:
    layer = BOFTLinear(16, 7, block_size=4, num_factors=3)
    inputs = torch.randn(2, 5, 16)
    torch.testing.assert_close(
        layer(inputs),
        F.linear(inputs, layer.base.weight, layer.base.bias),
    )

    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)
    rotations = layer.orthogonal_blocks()
    products = rotations.transpose(-2, -1) @ rotations
    assert torch.allclose(products, torch.eye(4), atol=1e-5, rtol=1e-5)

    dense = layer.dense_orthogonal_matrix()
    assert torch.allclose(
        dense.transpose(0, 1) @ dense,
        torch.eye(16),
        atol=2e-5,
        rtol=2e-5,
    )
    expected = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    assert torch.allclose(layer(inputs), expected, atol=1e-5, rtol=1e-5)


def test_vit_boft_model_and_optimizer() -> None:
    num_classes = 10
    model = ViTBOFTClassifier(num_classes=num_classes)
    assert load_pretrained_vit(model) == 198

    independent_entries = BOFT_BLOCK_SIZE * (BOFT_BLOCK_SIZE - 1) // 2
    boft_parameters = (
        12
        * 2
        * BOFT_NUM_FACTORS
        * (768 // BOFT_BLOCK_SIZE)
        * independent_entries
    )
    classifier_parameters = 768 * num_classes + num_classes
    counts = count_parameters(model)
    assert counts["boft"] == boft_parameters == 1_271_808
    assert counts["classifier"] == classifier_parameters
    assert counts["trainable"] == boft_parameters + classifier_parameters

    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        BOFT_LEARNING_RATE,
        CLASSIFIER_LEARNING_RATE,
    ]
    scheduler = build_scheduler(optimizer, num_epochs=100, warmup_epochs=10)
    assert scheduler is not None

    layers = model._cache_all_boft_rotations()
    try:
        assert len(layers) == 24
        loss = sum(layer.orthogonal_blocks().sum() for layer in layers)
        loss.backward()
    finally:
        for layer in layers:
            layer.clear_rotation_cache()
    assert all(layer.skew_parameters.grad is not None for layer in layers)


if __name__ == "__main__":
    test_boft_math()
    test_vit_boft_model_and_optimizer()
    print("Mini-VTAB BOFT tests passed.")
