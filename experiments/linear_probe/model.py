
"""ViT-B/16 linear probe with a frozen pretrained backbone."""

from experiments.linear_probe.config import NUM_CLASSES
from vit_core.vit_backbone import (
    VIT_BASE_PATCH16_224,
    ViTForImageClassification,
)


class ViTLinearProbe(ViTForImageClassification):
    """Train only a new classifier on top of frozen ViT features."""

    def __init__(self) -> None:
        super().__init__(
            num_classes=NUM_CLASSES,
            config=VIT_BASE_PATCH16_224,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Freeze every parameter except the randomly initialized classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
