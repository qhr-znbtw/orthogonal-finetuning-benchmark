"""下载本项目需要的ViT-B/16预训练文件。"""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "google/vit-base-patch16-224"
MODEL_DIR = Path(__file__).parent / "pretrained" / "vit-base-patch16-224"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "README.md",
        ],
    )

    print(f"Model downloaded to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
