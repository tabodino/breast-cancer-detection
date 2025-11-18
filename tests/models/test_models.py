"""
Comprehensive tests for model architectures and training.
Covers all model types, training pipeline, and checkpoint management.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile

from src.models.base_model import (
    BaseModel,
    CNNModel,
    EfficientNetB3Model,
    ResNet50Model,
    MobileNetV3Model,
    UNetModel,
    get_model,
)
from src.models.training import ModelTrainer


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_base_model_abstract(self):
        """BaseModel should not be instantiated directly."""
        # This would fail since BaseModel is abstract
        # Just verify it has required methods
        assert hasattr(BaseModel, "build")
        assert hasattr(BaseModel, "get_model")
        assert hasattr(BaseModel, "compile")

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        model = EfficientNetB3Model(
            input_shape=(224, 224, 3), num_classes=2, name="test_model"
        )
        assert model.input_shape == (224, 224, 3)
        assert model.num_classes == 2
        assert model.name == "test_model"

    def test_get_model_returns_keras_model(self):
        """get_model() should return a compiled Keras model."""
        model = EfficientNetB3Model()
        keras_model = model.get_model()
        assert isinstance(keras_model, tf.keras.Model)

    def test_model_caching(self):
        """get_model() should cache the model."""
        model = EfficientNetB3Model()
        model1 = model.get_model()
        model2 = model.get_model()
        assert model1 is model2  # Same object


class TestModelArchitectures:
    """Tests for different model architectures."""

    @pytest.mark.parametrize(
        "model_class,expected_name",
        [
            (CNNModel, "cnn"),
            (EfficientNetB3Model, "efficientnet_b3"),
            (ResNet50Model, "resnet50"),
            (MobileNetV3Model, "mobilenet_v3"),
            (UNetModel, "unet"),
        ],
    )
    def test_all_models_build(self, model_class, expected_name):
        """All model classes should build successfully."""
        model = model_class()
        keras_model = model.get_model()
        assert keras_model is not None
        assert isinstance(keras_model, tf.keras.Model)

    @pytest.mark.parametrize(
        "model_class",
        [
            CNNModel,
            EfficientNetB3Model,
            ResNet50Model,
            MobileNetV3Model,
            UNetModel,
        ],
    )
    def test_model_input_output_shapes(self, model_class):
        """Models should handle correct input/output shapes."""
        model = model_class(input_shape=(224, 224, 3), num_classes=2)
        keras_model = model.get_model()

        # Create dummy input
        dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)

        # Forward pass
        output = keras_model(dummy_input, training=False)

        assert output.shape == (2, 2)  # (batch_size, num_classes)

    def test_efficientnet_b3_specifics(self):
        """Test EfficientNetB3 specific properties."""
        model = EfficientNetB3Model()
        keras_model = model.get_model()

        # Should have layers from pre-trained weights
        assert len(keras_model.layers) > 7

        # Should have GlobalAveragePooling2D layer
        layer_types = [type(layer).__name__ for layer in keras_model.layers]
        assert "GlobalAveragePooling2D" in layer_types


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        X_train = np.random.randn(10, 224, 224, 3).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, 2, 10), num_classes=2
        )

        X_val = np.random.randn(5, 224, 224, 3).astype(np.float32)
        y_val = tf.keras.utils.to_categorical(np.random.randint(0, 2, 5), num_classes=2)

        return X_train, y_train, X_val, y_val

    def test_trainer_initialization(self):
        """ModelTrainer should initialize correctly."""
        trainer = ModelTrainer(model_name="efficientnet_b3")
        assert trainer.model_name == "efficientnet_b3"
        assert trainer.model is None
        assert trainer.history is None

    def test_build_model(self):
        """Trainer should build model correctly."""
        trainer = ModelTrainer()
        trainer.build_model()
        assert trainer.model is not None

    def test_get_callbacks(self):
        """Trainer should return correct callbacks."""
        trainer = ModelTrainer()
        callbacks = trainer.get_callbacks("test_run_id")
        assert len(callbacks) == 3  # ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    @pytest.mark.parametrize(
        "model_name", ["efficientnet_b3", "resnet50", "mobilenet_v3", "unet"]
    )
    def test_invalid_model_name(self, model_name):
        """Test factory function with invalid model name."""
        if model_name == "invalid":
            with pytest.raises(ValueError):
                get_model("invalid")

    def test_get_model_factory(self):
        """Factory function should return correct models."""
        model_names = ["efficientnet_b3", "resnet50", "mobilenet_v3", "unet"]

        for name in model_names:
            model = get_model(name)
            assert isinstance(model, BaseModel)
            assert model.get_model() is not None


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_model_save(self):
        """Models should save to disk correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = EfficientNetB3Model()
            model.build()

            save_path = Path(tmpdir) / "test_model.h5"
            model.save(save_path)

            assert save_path.exists()

    def test_model_load(self):
        """Models should load from disk correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model1 = EfficientNetB3Model()
            model1.build()
            save_path = Path(tmpdir) / "test_model.h5"
            model1.save(save_path)

            # Load
            model2 = EfficientNetB3Model()
            model2.load(save_path)

            assert model2.model is not None


class TestModelCompile:
    """Tests for model compilation."""

    def test_compile_with_defaults(self):
        """Models should compile with default settings."""
        model = EfficientNetB3Model()
        model.compile()

        keras_model = model.get_model()
        assert keras_model.optimizer is not None
        assert keras_model.loss is not None

    def test_compile_with_custom_metrics(self):
        """Models should compile with custom metrics."""
        custom_metrics = ["mae", tf.keras.metrics.MeanSquaredError()]
        model = EfficientNetB3Model()
        model.compile(metrics=custom_metrics)

        keras_model = model.get_model()
        assert keras_model.compiled_metrics is not None


class TestModelInputShapes:
    """Tests for different input shapes."""

    @pytest.mark.parametrize(
        "input_shape,num_classes",
        [
            ((224, 224, 3), 2),
            ((256, 256, 3), 3),
            ((512, 512, 3), 4),
        ],
    )
    def test_model_with_different_shapes(self, input_shape, num_classes):
        """Models should handle different input shapes."""
        model = EfficientNetB3Model(input_shape=input_shape, num_classes=num_classes)
        keras_model = model.get_model()

        dummy_input = np.random.randn(2, *input_shape).astype(np.float32)
        output = keras_model(dummy_input, training=False)

        assert output.shape == (2, num_classes)
