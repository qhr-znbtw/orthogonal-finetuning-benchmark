"""Tests for block-diagonal OFT and its Mini-VTAB ViT integration."""

import torch
from torch.nn import functional as F

from experiments.oft.config import (
    CLASSIFIER_LEARNING_RATE,
    OFT_BLOCK_SIZE,
    OFT_LEARNING_RATE,
)
from experiments.oft.model import ViTOFTClassifier
from experiments.oft.oft_layer import BlockOFTLinear
from experiments.oft.run_oft import build_optimizer, build_scheduler, count_parameters
from vit_core.pretrained import load_pretrained_vit


def test_oft_blocks_are_orthogonal() -> None:
    layer = BlockOFTLinear(16, 10, block_size=4)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotations = layer.orthogonal_blocks()
    identity = torch.eye(layer.block_size)
    products = rotations.transpose(-2, -1) @ rotations
    assert rotations.shape == (1, 4, 4, 4)
    assert torch.allclose(products, identity, atol=1e-5, rtol=1e-5)


def test_oft_dense_matrix_and_forward() -> None:
    layer = BlockOFTLinear(16, 7, block_size=4)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotation = layer.dense_orthogonal_matrix()
    block_mask = torch.zeros_like(rotation, dtype=torch.bool)
    for start in range(0, layer.in_features, layer.block_size):
        block_mask[start : start + layer.block_size, start : start + layer.block_size] = True

    assert torch.count_nonzero(rotation.masked_select(~block_mask)) == 0
    assert torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(layer.in_features),
        atol=2e-5,
        rtol=2e-5,
    )

    inputs = torch.randn(2, 5, 16)
    actual = layer(inputs)
    expected = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_vit_oft_model_optimizer_and_gradients() -> None:
    num_classes = 10
    model = ViTOFTClassifier(num_classes=num_classes)
    assert load_pretrained_vit(model) == 198

    independent_entries = OFT_BLOCK_SIZE * (OFT_BLOCK_SIZE - 1) // 2
    oft_parameters = 12 * 2 * (768 // OFT_BLOCK_SIZE) * independent_entries
    classifier_parameters = 768 * num_classes + num_classes
    counts = count_parameters(model)
    assert counts["oft"] == oft_parameters == 211_968
    assert counts["classifier"] == classifier_parameters
    assert counts["trainable"] == oft_parameters + classifier_parameters

    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        OFT_LEARNING_RATE,
        CLASSIFIER_LEARNING_RATE,
    ]
    scheduler = build_scheduler(optimizer, num_epochs=100, warmup_epochs=10)
    assert scheduler is not None

    layers = model._cache_all_oft_rotations()
    try:
        assert len(layers) == 24
        loss = sum(layer.orthogonal_blocks().sum() for layer in layers)
        loss.backward()
    finally:
        for layer in layers:
            layer.clear_rotation_cache()
    assert all(layer.skew_parameters.grad is not None for layer in layers)


if __name__ == "__main__":
    test_oft_blocks_are_orthogonal()
    test_oft_dense_matrix_and_forward()
    test_vit_oft_model_optimizer_and_gradients()
    print("Mini-VTAB block-OFT tests passed.")
