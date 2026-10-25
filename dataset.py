"""Oxford-IIIT Pet的Dataset划分与DataLoader构建。"""

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from net import build_image_transform


DATA_DIR = Path(__file__).parent / "data"
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_WORKERS = 4
VAL_RATIO = 0.2
RANDOM_SEED = 42


def stratified_split(
    labels: list[int],
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """按类别分别划分索引，使训练集与验证集保持类别平衡。"""
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_class[label].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in indices_by_class.values():
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[position] for position in permutation]
        val_size = round(len(indices) * val_ratio)

        val_indices.extend(shuffled_indices[:val_size])
        train_indices.extend(shuffled_indices[val_size:])

    return train_indices, val_indices


def build_datasets() -> tuple[Subset, Subset, OxfordIIITPet]:
    """构建训练集、验证集和官方测试集。"""
    index_source = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        download=False,
    )
    train_indices, val_indices = stratified_split(index_source._labels)

    train_source = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=build_image_transform(train=True),
        download=False,
    )
    val_source = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=build_image_transform(train=False),
        download=False,
    )
    test_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="category",
        transform=build_image_transform(train=False),
        download=False,
    )

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)
    return train_dataset, val_dataset, test_dataset


def build_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回训练、验证和测试DataLoader。"""
    train_dataset, val_dataset, test_dataset = build_datasets()
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()
    images, labels = next(iter(train_loader))

    print("Dataset sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Label range:", labels.min().item(), labels.max().item())
