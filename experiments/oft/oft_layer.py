
"""Block-diagonal OFT as the one-factor special case of BOFT."""

import torch

from experiments.boft.boft_layer import BOFTLinear


class BlockOFTLinear(BOFTLinear):
    """A frozen linear layer adapted by one block-diagonal orthogonal map."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            num_factors=1,
            bias=bias,
        )
        self._rotation_cache: torch.Tensor | None = None

    def orthogonal_blocks(self) -> torch.Tensor:
        """Return a model-level cached Cayley result when one is available."""
        if self._rotation_cache is not None:
            return self._rotation_cache
        return super().orthogonal_blocks()

    def set_rotation_cache(self, rotations: torch.Tensor) -> None:
        """Temporarily reuse rotations produced by one shared batched solve."""
        expected_shape = (1, self.num_blocks, self.block_size, self.block_size)
        if rotations.shape != expected_shape:
            raise ValueError(
                f"OFT rotation cache应为{expected_shape}，实际为{rotations.shape}。"
            )
        self._rotation_cache = rotations

    def clear_rotation_cache(self) -> None:
        """Drop the temporary graph-bearing cache after one model forward."""
        self._rotation_cache = None
