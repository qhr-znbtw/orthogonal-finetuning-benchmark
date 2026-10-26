
"""Fast end-to-end smoke test for the existing ViT-LoRA workflow."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.lora.model import ViTForPetClassification
from experiments.lora.run_lora import CHECKPOINT_PATH, build_optimizer
from vit_core.checkpoint import load_trainable_checkpoint
from vit_core.evaluation import test as evaluate
from vit_core.pet_dataset import build_datasets
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train_one_epoch


EXPECTED_TRAINABLE_PARAMETERS = 323_365
NUM_CLASSES = 37


def test_lora_workflow() -> None:
    """Load data, pretrained weights and the saved adapter, then run one image."""
    train_dataset, val_dataset, test_dataset = build_datasets()
    assert (len(train_dataset), len(val_dataset), len(test_dataset)) == (
        2_944,
        736,
        3_669,
    )

    model = ViTForPetClassification()
    load_pretrained_vit(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS
    assert model.head.classifier.out_features == NUM_CLASSES

    optimizer = build_optimizer(model)
    assert len(optimizer.param_groups) == 2

    checkpoint = load_trainable_checkpoint(model, CHECKPOINT_PATH)
    assert "model_state_dict" in checkpoint

    image, label = test_dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    one_sample_loader = DataLoader(
        TensorDataset(image.unsqueeze(0), torch.tensor([label])),
        batch_size=1,
    )
    criterion = nn.CrossEntropyLoss()
    train_metrics = train_one_epoch(
        model=model,
        dataloader=one_sample_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))

    eval_metrics = evaluate(
        model=model,
        dataloader=one_sample_loader,
        criterion=criterion,
        device=device,
    )

    assert eval_metrics["total"] == 1
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0


if __name__ == "__main__":
    test_lora_workflow()
    print("LoRA workflow smoke test passed.")
