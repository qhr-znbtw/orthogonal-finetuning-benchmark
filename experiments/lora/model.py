"""ViT-B/16 classifier with LoRA adapters in Query and Value projections."""

from functools import partial

import torch

from experiments.lora.config import LORA_ALPHA, LORA_RANK, NUM_CLASSES
from experiments.lora.lora_layer import LoRALinear
from vit_core.vit_backbone import (
    VIT_BASE_PATCH16_224,
    ViTForImageClassification,
)


IMAGE_SIZE = VIT_BASE_PATCH16_224.image_size
NUM_LAYERS = VIT_BASE_PATCH16_224.num_layers


class ViTForPetClassification(ViTForImageClassification):
    """用于Oxford-IIIT Pet分类的完整ViT-B/16 LoRA网络。"""

    def __init__(self) -> None:
        lora_factory = partial(
            LoRALinear,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
        )
        super().__init__(
            num_classes=NUM_CLASSES,
            config=VIT_BASE_PATCH16_224,
            query_factory=lora_factory,
            value_factory=lora_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        """只训练所有LoRA分支和新的37类分类器。"""
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


if __name__ == "__main__":
    example_batch = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    model = ViTForPetClassification()

    with torch.no_grad():
        logits = model(example_batch)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Logits shape:", logits.shape)  # torch.Size([1, 37])
    print("Trainable parameters:", trainable_parameters)
