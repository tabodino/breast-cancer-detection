from unittest.mock import patch
import pandas as pd
import numpy as np
from src.etl.preprocess_images_split import preprocess_split


def test_preprocess_split(monkeypatch, tmp_path):
    img_path = tmp_path / "img1.jpg"
    img_path.touch()
    df = pd.DataFrame([{"path": str(img_path), "label": 2}])
    index_csv = tmp_path / "df.csv"
    df.to_csv(index_csv, index=False)

    # Mock cv2.imread
    fake_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    monkeypatch.setattr("cv2.imread", lambda path: fake_img.copy())

    # Mock cv2.resize
    monkeypatch.setattr(
        "cv2.resize", lambda img, sz: np.ones((224, 224, 3), dtype=np.uint8)
    )

    # Mock np.save
    saves = {}

    def fake_save(path, array):
        saves[str(path)] = array

    monkeypatch.setattr("numpy.save", fake_save)

    out_x = tmp_path / "X_test.npy"
    out_y = tmp_path / "y_test.npy"
    X, y = preprocess_split(index_csv, out_x, out_y, img_size=(224, 224))

    # We check mocked result output
    assert X.shape == (1, 224, 224, 3)
    assert y.shape == (1,)
    assert set(saves.keys()) == {str(out_x), str(out_y)}


def test_preprocess_split_image_not_found(monkeypatch, tmp_path):
    df = pd.DataFrame([{"path": str(tmp_path / "missing.jpg"), "label": 1}])
    index_csv = tmp_path / "df.csv"
    df.to_csv(index_csv, index=False)

    # Mock cv2.imread: return None (simulate image failed)
    monkeypatch.setattr("cv2.imread", lambda path: None)
    # Mock resize/save to avoid disk access
    monkeypatch.setattr("cv2.resize", lambda img, sz: None)
    monkeypatch.setattr("numpy.save", lambda path, arr: None)

    with patch("src.etl.preprocess_images_split.logger.warning") as mock_warn:
        X, y = preprocess_split(
            index_csv, tmp_path / "X.npy", tmp_path / "y.npy", img_size=(224, 224)
        )
        mock_warn.assert_called_once()
        assert "Cannot read image" in mock_warn.call_args[0][0]
    assert len(X) == 0
    assert len(y) == 0
