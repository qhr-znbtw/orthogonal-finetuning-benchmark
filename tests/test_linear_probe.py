"""Fast smoke test for the linear-probe experiment."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.linear_probe.model import ViTLinearProbe
from experiments.linear_probe.run_linear_probe import CHECKPOINT_PATH, build_optimizer
from vit_core.checkpoint import load_trainable_checkpoint
from vit_core.pet_dataset import build_datasets
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train_one_epoch


EXPECTED_TRAINABLE_PARAMETERS = 28_453


def test_linear_probe_workflow() -> None:
    _, _, test_dataset = build_datasets()
    model = ViTLinearProbe()
    load_pretrained_vit(model)
    model.set_trainable_parameters()

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names == {
        "head.classifier.weight",
        "head.classifier.bias",
    }
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == EXPECTED_TRAINABLE_PARAMETERS

    checkpoint = load_trainable_checkpoint(model, CHECKPOINT_PATH)
    assert checkpoint["epoch"] == 16

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
        optimizer=build_optimizer(model),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))


if __name__ == "__main__":
    test_linear_probe_workflow()
    print("Linear-probe workflow smoke test passed.")
