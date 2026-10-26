"""ViT-B/16 classifier with BOFT adapters in Query and Value."""

from functools import partial

from experiments.boft.boft_layer import BOFTLinear
from experiments.boft.config import (
    BOFT_BLOCK_SIZE,
    BOFT_NUM_FACTORS,
    NUM_CLASSES,
)
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTBOFTForPetClassification(ViTForImageClassification):
    """Freeze ViT-B/16 and train BOFT on all Query/Value projections."""

    def __init__(self) -> None:
        boft_factory = partial(
            BOFTLinear,
            block_size=BOFT_BLOCK_SIZE,
            num_factors=BOFT_NUM_FACTORS,
        )
        super().__init__(
            num_classes=NUM_CLASSES,
            config=VIT_BASE_PATCH16_224,
            query_factory=boft_factory,
            value_factory=boft_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Train only BOFT skew coordinates and the new classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, BOFTLinear):
                module.skew_parameters.requires_grad = True

        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
