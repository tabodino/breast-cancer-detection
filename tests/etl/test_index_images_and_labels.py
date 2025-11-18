import pandas as pd
import math
from src.etl.index_images_and_labels import index_images_and_labels


def test_index_images_and_labels(tmp_path):
    # Mock images
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.png"
    img1.touch()
    img2.touch()
    images_dir = tmp_path
    labels_dir = tmp_path

    # Label files : one with label, one without
    label_file = labels_dir / "img1.txt"
    label_file.write_text("1 0.5 0.5 1.0 1.0\n2 0.1 0.1 0.2 0.2")
    label_file2 = labels_dir / "img2.txt"
    label_file2.write_text("")

    out_csv = tmp_path / "index.csv"
    df = index_images_and_labels(images_dir, labels_dir, out_csv)
    # Assertions : 2 lines, correct labels
    assert len(df) == 2
    assert 1 in df["label"].values
    assert any(
        pd.isna(label) or (isinstance(label, float) and math.isnan(label))
        for label in df["label"]
    )
    assert out_csv.exists()
