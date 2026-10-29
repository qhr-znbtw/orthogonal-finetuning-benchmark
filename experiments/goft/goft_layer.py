"""Strict GOFT using a parallel tree of trainable Givens rotations."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _build_parallel_givens_schedule(
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tree-structured pairs whose total count is ``dimension - 1``.

    At stride ``s``, the representative coordinates of adjacent length-``s``
    groups are paired. An unpaired final group is carried into the next stage.
    This also handles dimensions, such as 768, that are not powers of two.
    """
    if dimension < 2:
        raise ValueError("GOFT requires at least two input features.")

    left_indices: list[int] = []
    right_indices: list[int] = []
    stage_offsets = [0]
    stride = 1

    while stride < dimension:
        for left in range(0, dimension, 2 * stride):
            right = left + stride
            if right < dimension:
                left_indices.append(left)
                right_indices.append(right)
        stage_offsets.append(len(left_indices))
        stride *= 2

    if len(left_indices) != dimension - 1:
        raise RuntimeError("The parallel Givens schedule must contain d - 1 pairs.")

    return (
        torch.tensor(left_indices, dtype=torch.long),
        torch.tensor(right_indices, dtype=torch.long),
        torch.tensor(stage_offsets, dtype=torch.long),
    )


class GOFTLinear(nn.Module):
    """Frozen linear layer preceded by a strict orthogonal GOFT transform.

    One angle is learned for every edge of a balanced binary tree over the
    input coordinates. The resulting ``d - 1`` Givens rotations are evaluated
    in ``ceil(log2(d))`` parallel stages.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = nn.Linear(in_features, out_features, bias=bias)

        left, right, offsets = _build_parallel_givens_schedule(in_features)
        self.register_buffer("left_indices", left, persistent=False)
        self.register_buffer("right_indices", right, persistent=False)
        self.register_buffer("stage_offsets", offsets, persistent=False)

        # theta = pi * tanh(raw_angle), matching the official GOFT mapping.
        # Zero initialization makes the adapter exactly the identity map.
        self.raw_angles = nn.Parameter(torch.zeros(in_features - 1))

    @property
    def num_stages(self) -> int:
        return self.stage_offsets.numel() - 1

    def angles(self) -> torch.Tensor:
        return math.pi * torch.tanh(self.raw_angles)

    def _stage_bounds(self, stage_index: int) -> tuple[int, int]:
        return (
            int(self.stage_offsets[stage_index].item()),
            int(self.stage_offsets[stage_index + 1].item()),
        )

    def rotate_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the GOFT rotation to row-vector features, stage by stage."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected final dimension {self.in_features}, got {inputs.shape[-1]}."
            )

        outputs = inputs
        all_angles = self.angles().to(dtype=inputs.dtype)
        for stage_index in range(self.num_stages):
            start, end = self._stage_bounds(stage_index)
            left = self.left_indices[start:end]
            right = self.right_indices[start:end]
            theta = all_angles[start:end]
            cosine = torch.cos(theta)
            sine = torch.sin(theta)

            left_values = outputs.index_select(-1, left)
            right_values = outputs.index_select(-1, right)
            rotated_left = left_values * cosine + right_values * sine
            rotated_right = -left_values * sine + right_values * cosine

            updated = outputs.clone()
            updated.index_copy_(-1, left, rotated_left)
            updated.index_copy_(-1, right, rotated_right)
            outputs = updated
        return outputs

    def dense_orthogonal_matrix(self) -> torch.Tensor:
        """Materialize the rotation for tests, inspection, and merging."""
        identity = torch.eye(
            self.in_features,
            dtype=self.raw_angles.dtype,
            device=self.raw_angles.device,
        )
        return self.rotate_features(identity)

    def effective_weight(self) -> torch.Tensor:
        """Return ``W R^T`` without a dense matrix multiplication.

        Feature rows are transformed as ``x R`` before the frozen linear map.
        Therefore the equivalent PyTorch weight is ``W R^T``. Applying the
        transposed sparse stages in reverse order costs O(d^2), rather than the
        O(d^3) cost of multiplying two dense d-by-d matrices.
        """
        weight = self.base.weight
        all_angles = self.angles().to(dtype=weight.dtype)

        for stage_index in reversed(range(self.num_stages)):
            start, end = self._stage_bounds(stage_index)
            left = self.left_indices[start:end]
            right = self.right_indices[start:end]
            theta = all_angles[start:end]
            cosine = torch.cos(theta).unsqueeze(0)
            sine = torch.sin(theta).unsqueeze(0)

            left_columns = weight.index_select(1, left)
            right_columns = weight.index_select(1, right)
            rotated_left = left_columns * cosine - right_columns * sine
            rotated_right = left_columns * sine + right_columns * cosine

            updated = weight.clone()
            updated.index_copy_(1, left, rotated_left)
            updated.index_copy_(1, right, rotated_right)
            weight = updated
        return weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.effective_weight(), self.base.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"angles={self.raw_angles.numel()}, stages={self.num_stages}, "
            f"bias={self.base.bias is not None}"
        )
