"""ViT-B/16 classifier with block-diagonal OFT in Query and Value."""

from functools import partial

import torch

from experiments.oft.config import OFT_BLOCK_SIZE
from experiments.oft.oft_layer import BlockOFTLinear
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTOFTClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train one block-OFT factor per Q/V projection."""

    def __init__(self, num_classes: int) -> None:
        oft_factory = partial(BlockOFTLinear, block_size=OFT_BLOCK_SIZE)
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=oft_factory,
            value_factory=oft_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Train only OFT skew coordinates and the task-specific classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, BlockOFTLinear):
                module.skew_parameters.requires_grad = True

        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _oft_layers(self) -> list[BlockOFTLinear]:
        return [
            module
            for module in self.modules()
            if isinstance(module, BlockOFTLinear)
        ]

    def _cache_all_oft_rotations(self) -> list[BlockOFTLinear]:
        """Evaluate all Q/V Cayley maps with one batched linear solve."""
        layers = self._oft_layers()
        if not layers:
            raise RuntimeError("The model has no BlockOFTLinear layers.")

        with torch.autocast(
            device_type=layers[0].skew_parameters.device.type,
            enabled=False,
        ):
            skew = torch.cat(
                [layer.skew_matrices().float() for layer in layers],
                dim=0,
            )
            identity = torch.eye(
                OFT_BLOCK_SIZE,
                dtype=skew.dtype,
                device=skew.device,
            )
            rotations = torch.linalg.solve(identity + skew, identity - skew)

        for layer, rotation in zip(layers, rotations.unbind(dim=0)):
            layer.set_rotation_cache(rotation.unsqueeze(0))
        return layers

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reuse the batched Cayley result throughout one model forward."""
        layers = self._cache_all_oft_rotations()
        try:
            return super().forward(images)
        finally:
            for layer in layers:
                layer.clear_rotation_cache()
