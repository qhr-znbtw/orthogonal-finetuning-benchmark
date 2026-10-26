"""Fast smoke test for the full fine-tuning experiment."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.full_finetune.model import ViTFullFineTune
from experiments.full_finetune.run_full_finetune import (
    CHECKPOINT_PATH,
    build_optimizer,
)
from vit_core.checkpoint import load_trainable_checkpoint
from vit_core.pet_dataset import build_datasets
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train_one_epoch


EXPECTED_TRAINABLE_PARAMETERS = 85_827_109


def test_full_finetune_workflow() -> None:
    _, _, test_dataset = build_datasets()
    model = ViTFullFineTune()
    load_pretrained_vit(model)
    model.set_trainable_parameters()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert total_parameters == EXPECTED_TRAINABLE_PARAMETERS
    assert trainable_parameters == total_parameters

    checkpoint = load_trainable_checkpoint(model, CHECKPOINT_PATH)
    assert checkpoint["epoch"] == 5

    optimizer = build_optimizer(model)
    assert len(optimizer.param_groups) == 2

    image, label = test_dataset[0]
    dataloader = DataLoader(
        TensorDataset(image.unsqueeze(0), torch.tensor([label])),
        batch_size=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))


if __name__ == "__main__":
    test_full_finetune_workflow()
    print("Full fine-tuning workflow smoke test passed.")
