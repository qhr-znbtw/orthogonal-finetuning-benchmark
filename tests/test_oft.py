
"""Tests for block-diagonal OFT and its ViT integration."""

import torch
from torch.nn import functional as F

from experiments.oft.model import ViTOFTForPetClassification
from experiments.oft.oft_layer import BlockOFTLinear
from experiments.oft.run_oft import build_optimizer, count_parameters
from vit_core.pretrained import load_pretrained_vit


EXPECTED_TOTAL_PARAMETERS = 86_039_077
EXPECTED_TRAINABLE_PARAMETERS = 240_421
EXPECTED_OFT_PARAMETERS = 211_968
EXPECTED_CLASSIFIER_PARAMETERS = 28_453


def test_oft_blocks_are_orthogonal() -> None:
    layer = BlockOFTLinear(16, 10, block_size=4)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotations = layer.orthogonal_blocks()
    identity = torch.eye(layer.block_size)
    products = rotations.transpose(-2, -1) @ rotations
    assert rotations.shape[0] == 1
    assert torch.allclose(products, identity, atol=1e-5, rtol=1e-5)


def test_oft_dense_matrix_is_block_diagonal() -> None:
    layer = BlockOFTLinear(16, 10, block_size=4)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotation = layer.dense_orthogonal_matrix()
    block_mask = torch.zeros_like(rotation, dtype=torch.bool)
    for start in range(0, layer.in_features, layer.block_size):
        block_mask[
            start : start + layer.block_size,
            start : start + layer.block_size,
        ] = True

    assert torch.count_nonzero(rotation.masked_select(~block_mask)) == 0
    assert torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(layer.in_features),
        atol=2e-5,
        rtol=2e-5,
    )


def test_oft_forward_matches_explicit_weight() -> None:
    layer = BlockOFTLinear(16, 7, block_size=4)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    inputs = torch.randn(2, 5, 16)
    actual = layer(inputs)
    expected = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_vit_oft_parameter_counts_and_pretrained_loading() -> None:
    model = ViTOFTForPetClassification()
    assert load_pretrained_vit(model) == 198
    counts = count_parameters(model)
    assert counts == {
        "total": EXPECTED_TOTAL_PARAMETERS,
        "trainable": EXPECTED_TRAINABLE_PARAMETERS,
        "oft": EXPECTED_OFT_PARAMETERS,
        "classifier": EXPECTED_CLASSIFIER_PARAMETERS,
    }
    optimizer = build_optimizer(model)
    assert len(optimizer.param_groups) == 2


def test_vit_oft_shared_cayley_solve_preserves_all_gradients() -> None:
    model = ViTOFTForPetClassification()
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
    test_oft_dense_matrix_is_block_diagonal()
    test_oft_forward_matches_explicit_weight()
    test_vit_oft_parameter_counts_and_pretrained_loading()
    test_vit_oft_shared_cayley_solve_preserves_all_gradients()
    print("Block OFT tests passed.")
