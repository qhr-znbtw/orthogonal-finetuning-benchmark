
"""ViT-B/16 with every parameter available for fine-tuning."""

from experiments.full_finetune.config import NUM_CLASSES
from vit_core.vit_backbone import (
    VIT_BASE_PATCH16_224,
    ViTForImageClassification,
)


class ViTFullFineTune(ViTForImageClassification):
    """Fine-tune the complete pretrained ViT and the new classifier."""

    def __init__(self) -> None:
        super().__init__(
            num_classes=NUM_CLASSES,
            config=VIT_BASE_PATCH16_224,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Make every model parameter trainable."""
        for parameter in self.parameters():
            parameter.requires_grad = True
