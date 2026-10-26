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
            f"第{factor_index + 1}个factor需要micro-block数量"
            f"{num_micro_blocks}能被{group_size}整除。"
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
    """Frozen linear layer preceded by a trainable orthogonal butterfly map.

    The adapted PyTorch weight is ``W' = W R^T``.  Instead of materializing
    the dense ``R``, the forward pass applies its sparse factors to the last
    feature dimension and then calls the frozen base linear layer.
    """

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
            raise ValueError("BOFT block_size必须是正偶数。")
        if in_features % block_size != 0:
            raise ValueError("in_features必须能被block_size整除。")
        if num_factors <= 0:
            raise ValueError("num_factors必须是正整数。")

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_factors = num_factors
        self.num_blocks = in_features // block_size

        # The base layer receives pretrained weights and remains frozen.
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # Store only the independent entries of each skew-symmetric matrix.
        # This represents exactly the same Q=A-A^T family without keeping the
        # symmetric part of A, whose gradient would always be zero.
        independent_entries = block_size * (block_size - 1) // 2
        self.skew_parameters = nn.Parameter(
            torch.zeros(
                num_factors,
                self.num_blocks,
                independent_entries,
            )
        )

        upper_row, upper_col = torch.triu_indices(
            block_size,
            block_size,
            offset=1,
        )
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

    def skew_matrices(self) -> torch.Tensor:
        """Build all trainable skew-symmetric block generators."""
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
        """Map every skew block to an orthogonal block with the Cayley map."""
        # Small matrix solves are kept in FP32 even during BF16 autocast.
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
            # solve(I + Q, I - Q) = (I + Q)^-1 (I - Q).
            # The two polynomial factors commute, so this is the Cayley form
            # (I - Q)(I + Q)^-1 without an explicit matrix inverse.
            return torch.linalg.solve(identity + skew, identity - skew)

    def rotate_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply all sparse butterfly factors to the final tensor axis."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"最后一维应为{self.in_features}，实际为{inputs.shape[-1]}。"
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
            permuted = rotated_blocks.flatten(-2)
            outputs = permuted.index_select(-1, inverse)

        return outputs

    def dense_orthogonal_matrix(self) -> torch.Tensor:
        """Materialize the dense R for tests and analysis, not training."""
        identity = torch.eye(
            self.in_features,
            dtype=self.skew_parameters.dtype,
            device=self.skew_parameters.device,
        )
        return self.rotate_features(identity)

    def effective_weight(self) -> torch.Tensor:
        """Return the explicit adapted weight ``W R^T`` for verification."""
        rotation = self.dense_orthogonal_matrix()
        return self.base.weight @ rotation.transpose(0, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rotated_inputs = self.rotate_features(inputs)
        return F.linear(rotated_inputs, self.base.weight, self.base.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, num_factors={self.num_factors}, "
            f"bias={self.base.bias is not None}"
        )
