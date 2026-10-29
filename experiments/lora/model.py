"""ViT-B/16 classifier with LoRA adapters in Query and Value."""

from functools import partial

from experiments.lora.config import LORA_ALPHA, LORA_RANK
from experiments.lora.lora_layer import LoRALinear
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTLoRAClassifier(ViTForImageClassification):
    """Train Query/Value LoRA branches and a task-specific classifier."""

    def __init__(self, num_classes: int) -> None:
        lora_factory = partial(
            LoRALinear,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
        )
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=lora_factory,
            value_factory=lora_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """Freeze the backbone and expose only LoRA plus classifier parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, LoRALinear):
                for parameter in module.lora_a.parameters():
                    parameter.requires_grad = True
                for parameter in module.lora_b.parameters():
                    parameter.requires_grad = True

        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
