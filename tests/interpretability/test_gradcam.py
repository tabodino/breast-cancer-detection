import pytest
import numpy as np
import tensorflow as tf


# Import functions to test
from src.interpretability import gradcam


@pytest.fixture
def dummy_image():
    # Create a dummy RGB image
    return np.ones((32, 32, 3), dtype=np.uint8) * 255


def test_preprocess_image(dummy_image):
    out = gradcam.preprocess_image(dummy_image, (32, 32))
    assert out.shape == (32, 32, 3)
    assert np.all(out <= 1.0)  # normalized


# --- Tests for gradcam_keras ---


def test_gradcam_keras_success(dummy_image):
    # Build a minimal Keras model with conv + dense
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(1, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Run Grad-CAM
    res = gradcam.gradcam_keras(model, dummy_image, target_size=(32, 32))

    # Assertions
    assert "overlay" in res
    assert "heatmap" in res
    assert isinstance(res["overlay"], np.ndarray)
    assert res["heatmap"].shape == (32, 32)


def test_gradcam_keras_no_conv_layer(dummy_image):
    class DummyModel:
        inputs = "x"
        layers = []

    with pytest.raises(ValueError):
        gradcam.gradcam_keras(DummyModel(), dummy_image, target_size=(32, 32))


def test_gradcam_keras_full_path(dummy_image):
    # Build a minimal Keras model with a conv layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(1, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Call gradcam_keras with this real model
    res = gradcam.gradcam_keras(model, dummy_image, target_size=(32, 32))

    # Assertions
    assert "overlay" in res
    assert "heatmap" in res
    assert isinstance(res["overlay"], np.ndarray)
    assert isinstance(res["heatmap"], np.ndarray)
    assert res["heatmap"].shape == (32, 32)
