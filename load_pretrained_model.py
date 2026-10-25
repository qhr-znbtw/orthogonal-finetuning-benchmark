"""把Google ViT-B/16预训练权重加载到本项目的LoRA模型中。"""

from pathlib import Path

import torch
from safetensors.torch import load_file

from net import NUM_LAYERS, ViTForPetClassification


WEIGHTS_PATH = (
    Path(__file__).parent
    / "pretrained"
    / "vit-base-patch16-224"
    / "model.safetensors"
)


def build_weight_map() -> dict[str, str]:
    """建立“我们的参数名 -> Hugging Face参数名”的对应关系。"""
    weight_map = {
        "embedding.cls_token": "vit.embeddings.cls_token",
        "embedding.position_embedding": "vit.embeddings.position_embeddings",
        "embedding.patch_embedding.projection.weight": (
            "vit.embeddings.patch_embeddings.projection.weight"
        ),
        "embedding.patch_embedding.projection.bias": (
            "vit.embeddings.patch_embeddings.projection.bias"
        ),
        "head.norm.weight": "vit.layernorm.weight",
        "head.norm.bias": "vit.layernorm.bias",
    }

    for index in range(NUM_LAYERS):
        ours = f"encoder.blocks.{index}"
        source = f"vit.encoder.layer.{index}"

        weight_map.update(
            {
                f"{ours}.norm1.weight": f"{source}.layernorm_before.weight",
                f"{ours}.norm1.bias": f"{source}.layernorm_before.bias",
                f"{ours}.attention.query.base.weight": (
                    f"{source}.attention.attention.query.weight"
                ),
                f"{ours}.attention.query.base.bias": (
                    f"{source}.attention.attention.query.bias"
                ),
                f"{ours}.attention.key.weight": (
                    f"{source}.attention.attention.key.weight"
                ),
                f"{ours}.attention.key.bias": (
                    f"{source}.attention.attention.key.bias"
                ),
                f"{ours}.attention.value.base.weight": (
                    f"{source}.attention.attention.value.weight"
                ),
                f"{ours}.attention.value.base.bias": (
                    f"{source}.attention.attention.value.bias"
                ),
                f"{ours}.attention.output.weight": (
                    f"{source}.attention.output.dense.weight"
                ),
                f"{ours}.attention.output.bias": (
                    f"{source}.attention.output.dense.bias"
                ),
                f"{ours}.norm2.weight": f"{source}.layernorm_after.weight",
                f"{ours}.norm2.bias": f"{source}.layernorm_after.bias",
                f"{ours}.mlp.fc1.weight": f"{source}.intermediate.dense.weight",
                f"{ours}.mlp.fc1.bias": f"{source}.intermediate.dense.bias",
                f"{ours}.mlp.fc2.weight": f"{source}.output.dense.weight",
                f"{ours}.mlp.fc2.bias": f"{source}.output.dense.bias",
            }
        )

    return weight_map


def load_pretrained_model() -> ViTForPetClassification:
    """返回已加载ViT预训练主干、且只开放LoRA和分类头的模型。"""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"找不到预训练权重：{WEIGHTS_PATH}\n"
            "请先运行 download_pretrained_model.py。"
        )

    model = ViTForPetClassification()
    source_state = load_file(WEIGHTS_PATH, device="cpu")
    weight_map = build_weight_map()

    mapped_state = {
        target_name: source_state[source_name]
        for target_name, source_name in weight_map.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(mapped_state, strict=False)

    expected_missing = {
        name
        for name in model.state_dict()
        if ".lora_a." in name
        or ".lora_b." in name
        or name.startswith("head.classifier.")
    }

    if set(missing_keys) != expected_missing or unexpected_keys:
        raise RuntimeError(
            "预训练权重没有按预期加载。\n"
            f"缺失参数：{missing_keys}\n"
            f"多余参数：{unexpected_keys}"
        )

    model.set_trainable_parameters()
    return model


if __name__ == "__main__":
    loaded_model = load_pretrained_model()
    loaded_model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in loaded_model.parameters()
        if parameter.requires_grad
    )

    with torch.no_grad():
        example_batch = torch.randn(1, 3, 224, 224)
        logits = loaded_model(example_batch)

    print(f"Loaded pretrained tensors: {len(build_weight_map())}")
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Logits shape: {logits.shape}")
