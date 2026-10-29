"""ViT-B/16 linear probe with a frozen pretrained backbone."""

from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTLinearProbe(ViTForImageClassification):
    """Train only a task-specific classifier on frozen ViT features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, config=VIT_BASE_PATCH16_224)
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
