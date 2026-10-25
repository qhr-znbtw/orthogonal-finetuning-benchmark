"""Download and verify the Oxford-IIIT Pet classification dataset."""

from pathlib import Path

from torchvision.datasets import OxfordIIITPet


def main() -> None:
    data_root = Path(__file__).resolve().parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_dataset = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        download=True,
    )
    test_dataset = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        download=True,
    )

    print(f"Dataset directory: {data_root}")
    print(f"Training/validation images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print("Oxford-IIIT Pet is ready.")


if __name__ == "__main__":
    main()
