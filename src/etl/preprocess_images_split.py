from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from loguru import logger
from src.config import get_settings
from tqdm import tqdm

settings = get_settings()


def preprocess_split(index_csv: Path, out_x: Path, out_y: Path, img_size=(224, 224)):
    df = pd.read_csv(index_csv)
    X, y = [], []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Preprocessing {index_csv.stem}"
    ):
        img = cv2.imread(row["path"])
        if img is None:
            logger.warning(f"Cannot read image: {row['path']}")
            continue
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        X.append(img)
        y.append(row["label"])
    X = np.array(X)
    y = np.array(y)
    np.save(out_x, X)
    np.save(out_y, y)
    logger.success(f"Saved {out_x} (shape={X.shape}), {out_y} (shape={y.shape})")
    return X, y


if __name__ == "__main__":
    splits = ["train", "valid", "test"]
    for split in splits:
        preprocess_split(
            Path(f"data/processed/{split}_index.csv"),
            Path(f"data/processed/X_{split}.npy"),
            Path(f"data/processed/y_{split}.npy"),
            img_size=(224, 224),
        )
