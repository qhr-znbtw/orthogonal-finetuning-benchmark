"""ViT-B/16 classifier with BOFT adapters in Query and Value."""

from functools import partial

import torch

from experiments.boft.boft_layer import BOFTLinear
from experiments.boft.config import BOFT_BLOCK_SIZE, BOFT_NUM_FACTORS
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTBOFTClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train six-factor BOFT on all Q/V projections."""

    def __init__(self, num_classes: int) -> None:
        boft_factory = partial(
            BOFTLinear,
            block_size=BOFT_BLOCK_SIZE,
            num_factors=BOFT_NUM_FACTORS,
        )
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=boft_factory,
            value_factory=boft_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Train only BOFT skew coordinates and the task-specific classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, BOFTLinear):
                module.skew_parameters.requires_grad = True

        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _boft_layers(self) -> list[BOFTLinear]:
        return [module for module in self.modules() if isinstance(module, BOFTLinear)]

    def _cache_all_boft_rotations(self) -> list[BOFTLinear]:
        """Evaluate all Q/V Cayley maps with one batched linear solve."""
        layers = self._boft_layers()
        if not layers:
            raise RuntimeError("The model has no BOFTLinear layers.")

        with torch.autocast(
            device_type=layers[0].skew_parameters.device.type,
            enabled=False,
        ):
            skew = torch.cat(
                [layer.skew_matrices().float() for layer in layers],
                dim=0,
            )
            identity = torch.eye(
                BOFT_BLOCK_SIZE,
                dtype=skew.dtype,
                device=skew.device,
            )
            rotations = torch.linalg.solve(identity + skew, identity - skew)

        chunks = rotations.split(BOFT_NUM_FACTORS, dim=0)
        for layer, rotation in zip(layers, chunks):
            layer.set_rotation_cache(rotation)
        return layers

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reuse the batched Cayley result throughout one model forward."""
        layers = self._cache_all_boft_rotations()
        try:
            return super().forward(images)
        finally:
            for layer in layers:
                layer.clear_rotation_cache()
