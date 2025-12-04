import pytest
import numpy as np

from src.interpretability import shap_utils


@pytest.fixture
def dummy_image():
    return np.ones((64, 64, 3), dtype=np.uint8) * 255


@pytest.fixture
def dummy_images():
    return [np.ones((64, 64, 3), dtype=np.uint8) * i for i in range(1, 4)]


# --- Tests for preprocess_batch ---


def test_preprocess_batch(dummy_images):
    out = shap_utils.preprocess_batch(dummy_images, (32, 32))
    assert out.shape == (3, 32, 32, 3)
    assert np.all(out <= 1.0)


# --- Tests for shap_keras ---


def test_shap_keras(monkeypatch, dummy_image, dummy_images):
    class DummyExplainer:
        def shap_values(self, x, nsamples=100):
            return [np.ones((1, 32, 32, 3))]

    monkeypatch.setattr("shap.GradientExplainer", lambda model, bg: DummyExplainer())

    class DummyModel:
        pass

    res = shap_utils.shap_keras(
        DummyModel(), dummy_image, dummy_images, target_size=(32, 32)
    )
    assert isinstance(res, list)
    assert res[0].shape == (1, 32, 32, 3)
