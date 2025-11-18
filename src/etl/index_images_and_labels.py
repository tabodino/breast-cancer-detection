from pathlib import Path
import pandas as pd
from loguru import logger
from src.config import get_settings

settings = get_settings()


def index_images_and_labels(images_dir: Path, labels_dir: Path, out_csv: Path):
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    data = []
    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        label = None
        if label_path.exists():
            with open(label_path) as f:
                labels = [int(line.split()[0]) for line in f if line.strip()]
                label = labels[0] if labels else None
        data.append({"path": str(img_path.resolve()), "label": label})
    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)
    logger.success(f"Indexed {len(df)} images and saved to {out_csv}")
    return df


if __name__ == "__main__":
    splits = ["train", "valid", "test"]
    for split in splits:
        index_images_and_labels(
            Path(f"data/raw/{split}/images"),
            Path(f"data/raw/{split}/labels"),
            Path(f"data/processed/{split}_index.csv"),
        )
