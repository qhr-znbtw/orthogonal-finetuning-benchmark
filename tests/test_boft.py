"""Tests for the BOFT parameterization and ViT integration."""

import torch
from torch.nn import functional as F

from experiments.boft.boft_layer import BOFTLinear
from experiments.boft.model import ViTBOFTForPetClassification
from experiments.boft.run_boft import build_optimizer, count_parameters
from vit_core.pretrained import load_pretrained_vit


EXPECTED_TOTAL_PARAMETERS = 87_098_917
EXPECTED_TRAINABLE_PARAMETERS = 1_300_261
EXPECTED_BOFT_PARAMETERS = 1_271_808
EXPECTED_CLASSIFIER_PARAMETERS = 28_453


def test_cayley_blocks_are_orthogonal() -> None:
    layer = BOFTLinear(16, 10, block_size=4, num_factors=3)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotations = layer.orthogonal_blocks()
    identity = torch.eye(layer.block_size)
    products = rotations.transpose(-2, -1) @ rotations
    assert torch.allclose(products, identity, atol=1e-5, rtol=1e-5)


def test_dense_butterfly_is_orthogonal() -> None:
    layer = BOFTLinear(16, 10, block_size=4, num_factors=3)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    rotation = layer.dense_orthogonal_matrix()
    identity = torch.eye(layer.in_features)
    assert torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )


def test_forward_matches_explicit_adapted_weight() -> None:
    layer = BOFTLinear(16, 7, block_size=4, num_factors=3)
    with torch.no_grad():
        layer.skew_parameters.normal_(mean=0.0, std=0.1)

    inputs = torch.randn(2, 5, 16)
    actual = layer(inputs)
    expected = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_zero_initialization_preserves_base_layer() -> None:
    layer = BOFTLinear(16, 7, block_size=4, num_factors=3)
    inputs = torch.randn(2, 5, 16)
    expected = F.linear(inputs, layer.base.weight, layer.base.bias)
    assert torch.allclose(layer(inputs), expected, atol=1e-6, rtol=1e-6)


def test_vit_boft_parameter_counts_and_pretrained_loading() -> None:
    model = ViTBOFTForPetClassification()
    assert load_pretrained_vit(model) == 198
    counts = count_parameters(model)
    assert counts == {
        "total": EXPECTED_TOTAL_PARAMETERS,
        "trainable": EXPECTED_TRAINABLE_PARAMETERS,
        "boft": EXPECTED_BOFT_PARAMETERS,
        "classifier": EXPECTED_CLASSIFIER_PARAMETERS,
    }
    optimizer = build_optimizer(model)
    assert len(optimizer.param_groups) == 2


if __name__ == "__main__":
    test_cayley_blocks_are_orthogonal()
    test_dense_butterfly_is_orthogonal()
    test_forward_matches_explicit_adapted_weight()
    test_zero_initialization_preserves_base_layer()
    test_vit_boft_parameter_counts_and_pretrained_loading()
    print("BOFT tests passed.")
