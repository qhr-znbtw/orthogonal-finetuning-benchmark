"""A transparent block-butterfly orthogonal linear adapter."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _build_coordinate_permutation(
    in_features: int,
    block_size: int,
    factor_index: int,
) -> torch.Tensor:
    """Return the fixed coordinate order for one binary butterfly factor."""
    micro_block_size = block_size // 2
    num_micro_blocks = in_features // micro_block_size
    stride = 2**factor_index
    group_size = 2 * stride

    if num_micro_blocks % group_size != 0:
        raise ValueError(
            f"Factor {factor_index + 1} requires {num_micro_blocks} "
            f"micro-blocks to be divisible by {group_size}."
        )

    micro_block_order: list[int] = []
    for group_start in range(0, num_micro_blocks, group_size):
        for offset in range(stride):
            micro_block_order.extend(
                (group_start + offset, group_start + offset + stride)
            )

    coordinate_order: list[int] = []
    for micro_block in micro_block_order:
        coordinate_start = micro_block * micro_block_size
        coordinate_order.extend(
            range(coordinate_start, coordinate_start + micro_block_size)
        )
    return torch.tensor(coordinate_order, dtype=torch.long)


class BOFTLinear(nn.Module):
    """Frozen linear layer preceded by trainable orthogonal butterfly maps."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int,
        num_factors: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("BOFT block_size must be a positive even integer.")
        if in_features % block_size != 0:
            raise ValueError("in_features must be divisible by block_size.")
        if num_factors <= 0:
            raise ValueError("num_factors must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_factors = num_factors
        self.num_blocks = in_features // block_size
        self.base = nn.Linear(in_features, out_features, bias=bias)

        independent_entries = block_size * (block_size - 1) // 2
        self.skew_parameters = nn.Parameter(
            torch.zeros(num_factors, self.num_blocks, independent_entries)
        )

        upper_row, upper_col = torch.triu_indices(block_size, block_size, offset=1)
        self.register_buffer("upper_row", upper_row, persistent=False)
        self.register_buffer("upper_col", upper_col, persistent=False)

        permutations = []
        inverse_permutations = []
        for factor_index in range(num_factors):
            order = _build_coordinate_permutation(
                in_features,
                block_size,
                factor_index,
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(in_features)
            permutations.append(order)
            inverse_permutations.append(inverse)

        self.register_buffer(
            "permutations",
            torch.stack(permutations),
            persistent=False,
        )
        self.register_buffer(
            "inverse_permutations",
            torch.stack(inverse_permutations),
            persistent=False,
        )
        self._rotation_cache: torch.Tensor | None = None

    def skew_matrices(self) -> torch.Tensor:
        """Build all skew-symmetric block generators."""
        shape = (
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        skew = self.skew_parameters.new_zeros(shape)
        skew[..., self.upper_row, self.upper_col] = self.skew_parameters
        skew[..., self.upper_col, self.upper_row] = -self.skew_parameters
        return skew

    def orthogonal_blocks(self) -> torch.Tensor:
        """Map every skew block through the Cayley transform."""
        if self._rotation_cache is not None:
            return self._rotation_cache
        with torch.autocast(
            device_type=self.skew_parameters.device.type,
            enabled=False,
        ):
            skew = self.skew_matrices().float()
            identity = torch.eye(
                self.block_size,
                dtype=skew.dtype,
                device=skew.device,
            )
            return torch.linalg.solve(identity + skew, identity - skew)

    def set_rotation_cache(self, rotations: torch.Tensor) -> None:
        """Temporarily reuse rotations from one model-level batched solve."""
        expected_shape = (
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if rotations.shape != expected_shape:
            raise ValueError(
                f"BOFT rotation cache must have shape {expected_shape}, "
                f"got {tuple(rotations.shape)}."
            )
        self._rotation_cache = rotations

    def clear_rotation_cache(self) -> None:
        self._rotation_cache = None

    def rotate_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply all sparse butterfly factors to the final tensor axis."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected final dimension {self.in_features}, got {inputs.shape[-1]}."
            )

        rotations = self.orthogonal_blocks().to(dtype=inputs.dtype)
        outputs = inputs
        for factor_index in range(self.num_factors):
            order = self.permutations[factor_index]
            inverse = self.inverse_permutations[factor_index]
            permuted = outputs.index_select(-1, order)
            blocks = permuted.reshape(
                *permuted.shape[:-1],
                self.num_blocks,
                self.block_size,
            )
            rotated_blocks = torch.einsum(
                "...ki,kij->...kj",
                blocks,
                rotations[factor_index],
            )
            outputs = rotated_blocks.flatten(-2).index_select(-1, inverse)
        return outputs

    def dense_orthogonal_matrix(self) -> torch.Tensor:
        identity = torch.eye(
            self.in_features,
            dtype=self.skew_parameters.dtype,
            device=self.skew_parameters.device,
        )
        return self.rotate_features(identity)

    def effective_weight(self) -> torch.Tensor:
        rotation = self.dense_orthogonal_matrix()
        return self.base.weight @ rotation.transpose(0, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(self.rotate_features(inputs), self.base.weight, self.base.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, num_factors={self.num_factors}, "
            f"bias={self.base.bias is not None}"
        )
