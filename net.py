"""ViT-B/16 的输入嵌入与 Transformer Encoder Block。"""

import math

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 12
LORA_RANK = 8
LORA_ALPHA = 8
NUM_CLASSES = 37
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def build_image_transform(train: bool) -> transforms.Compose:
    """创建训练集或验证集的图像预处理。"""
    if train:
        resize_or_crop = [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        resize_or_crop = [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            )
        ]

    return transforms.Compose(
        resize_or_crop
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


class PatchEmbedding(nn.Module):
    """把 [B, 3, 224, 224] 的图片转换为 [B, 196, 768] 的 patch tokens。"""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(images)       # [B, 768, 14, 14]
        tokens = tokens.flatten(2)             # [B, 768, 196]
        tokens = tokens.transpose(1, 2)        # [B, 196, 768]
        return tokens


class ViTInputEmbedding(nn.Module):
    """生成送入第一个 Transformer Block 的 token 序列。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding()

        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, HIDDEN_SIZE)
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embedding(images)  # [B, 196, 768]

        batch_size = images.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)  # [B, 197, 768]

        return tokens + self.position_embedding


class LoRALinear(nn.Module):
    """在线性层旁增加一个低秩更新：Wx + (alpha/r)BAx。"""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Linear(in_features, LORA_RANK, bias=False)
        self.lora_b = nn.Linear(LORA_RANK, out_features, bias=False)
        self.scaling = LORA_ALPHA / LORA_RANK

        # B从零开始，使LoRA在训练开始时不会改变预训练模型的输出。
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        lora_output = self.lora_b(self.lora_a(inputs))
        return base_output + self.scaling * lora_output


class MultiHeadSelfAttention(nn.Module):
    """ViT-B/16 的多头自注意力。"""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_size = HIDDEN_SIZE // NUM_HEADS

        # LoRA只加入Query和Value；Key与输出投影保持原结构。
        self.query = LoRALinear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.key = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.value = LoRALinear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.reshape(
            batch_size, num_tokens, self.num_heads, self.head_size
        )
        return tensor.transpose(1, 2)  # [B, 12, 197, 64]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(tokens))
        value = self._split_heads(self.value(tokens))

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_weights = attention_scores.softmax(dim=-1)

        context = attention_weights @ value             # [B, 12, 197, 64]
        context = context.transpose(1, 2).contiguous()   # [B, 197, 12, 64]
        context = context.flatten(2)                     # [B, 197, 768]
        return self.output(context)


class MLP(nn.Module):
    """逐个处理token的前馈网络：768 -> 3072 -> 768。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.fc1(tokens)
        tokens = self.activation(tokens)
        return self.fc2(tokens)


class TransformerEncoderBlock(nn.Module):
    """一个Pre-LayerNorm结构的ViT Transformer Encoder Block。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.attention = MultiHeadSelfAttention()
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.mlp = MLP()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class TransformerEncoder(nn.Module):
    """ViT-Base的完整Encoder：顺序堆叠12个Transformer Encoder Block。"""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock() for _ in range(NUM_LAYERS)]
        )
        self.freeze_base_parameters()

    def freeze_base_parameters(self) -> None:
        """冻结原Encoder，只保留所有Query/Value中的LoRA参数可训练。"""
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in self.modules():
            if isinstance(module, LoRALinear):
                for parameter in module.lora_a.parameters():
                    parameter.requires_grad = True
                for parameter in module.lora_b.parameters():
                    parameter.requires_grad = True

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class ClassificationHead(nn.Module):
    """对最终CLS表示做LayerNorm，再输出37个宠物类别的logits。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(tokens)
        cls_token = tokens[:, 0]
        return self.classifier(cls_token)


class ViTForPetClassification(nn.Module):
    """用于Oxford-IIIT Pet分类的完整ViT-B/16 LoRA网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = ViTInputEmbedding()
        self.encoder = TransformerEncoder()
        self.head = ClassificationHead()
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(images)  # [B, 3, 224, 224] -> [B, 197, 768]
        tokens = self.encoder(tokens)     # [B, 197, 768] -> [B, 197, 768]
        return self.head(tokens)          # [B, 197, 768] -> [B, 37]


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
