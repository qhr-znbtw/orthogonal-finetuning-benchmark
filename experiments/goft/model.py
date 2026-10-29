"""ViT-B/16 classifier with strict GOFT adapters in Query and Value."""

from experiments.goft.goft_layer import GOFTLinear
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTGOFTClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train strict GOFT on all Q/V projections."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=GOFTLinear,
            value_factory=GOFTLinear,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Train only GOFT angles and the task-specific classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, GOFTLinear):
                module.raw_angles.requires_grad = True

        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
