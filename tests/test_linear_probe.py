"""Smoke tests for the Mini-VTAB frozen-backbone baseline."""

import torch

from experiments.linear_probe.model import ViTLinearProbe
from experiments.linear_probe.run_linear_probe import build_optimizer
from vit_core.mini_vtab import TASK_SPECS, build_datasets
from vit_core.pretrained import load_pretrained_vit


def test_pretrained_linear_probe() -> None:
    model = ViTLinearProbe(num_classes=47)
    assert load_pretrained_vit(model) == 198
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"head.classifier.weight", "head.classifier.bias"}
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 36_143
    assert len(build_optimizer(model).param_groups) == 1

    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 47)


def test_dataset_metadata() -> None:
    assert [TASK_SPECS[name].num_classes for name in TASK_SPECS] == [47, 10, 16]
    for task_name, expected_test_size in (("dtd", 1880), ("eurosat", 5400)):
        train_dataset, val_dataset, test_dataset = build_datasets(task_name)
        assert len(train_dataset) == 800
        assert len(val_dataset) == 200
        assert len(test_dataset) == expected_test_size
        image, label = train_dataset[0]
        assert image.shape == (3, 224, 224)
        assert 0 <= label < TASK_SPECS[task_name].num_classes


if __name__ == "__main__":
    test_pretrained_linear_probe()
    test_dataset_metadata()
    print("Mini-VTAB linear-probe smoke tests passed.")
