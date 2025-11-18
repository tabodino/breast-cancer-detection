from tensorflow import keras
from tensorflow.keras import layers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from loguru import logger


class BaseModel(ABC):
    """Abstract base class for all model architectures."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1,
        name: str = "BaseModel",
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = None
        self.history = None

    @abstractmethod
    def build(self) -> keras.Model:
        """Build and return the model architecture."""
        pass

    def get_model(self) -> keras.Model:
        """Get or build the model."""
        if self.model is None:
            self.model = self.build()
        return self.model

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: list = None,
    ):
        """Compile the model."""

        if metrics is None:
            metrics = [
                "accuracy",
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC(),
            ]

        self.get_model().compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info(f"Model {self.name} compiled successfully")

    def save(self, save_path: Path):
        """Save model to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_model().save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, load_path: Path):
        """Load model from disk."""
        self.model = keras.models.load_model(load_path)
        logger.info(f"Model loaded from {load_path}")

    def summary(self):
        """Print model summary."""
        return self.get_model().summary()

    def _build_output_layer(self, x):
        """Build output layer based on num_classes.

        Args:
            x: Input tensor from previous layer

        Returns:
            Output tensor with appropriate activation
        """
        if self.num_classes == 1:
            return layers.Dense(1, activation="sigmoid")(x)
        else:
            return layers.Dense(self.num_classes, activation="softmax")(x)


class CNNModel(BaseModel):
    """A simple Convolutional Neural Network model."""

    def build(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

        outputs = self._build_output_layer(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class EfficientNetB3Model(BaseModel):
    """EfficientNetB3 with custom classification head."""

    def build(self) -> keras.Model:
        """Build EfficientNetB3 architecture."""

        # Load pre-trained EfficientNetB3
        base_model = keras.applications.EfficientNetB3(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )

        # Freeze base model layers
        base_model.trainable = True

        # Fine-tune only the last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Build custom head
        inputs = keras.Input(shape=self.input_shape)

        # Preprocess input for EfficientNet
        x = keras.applications.efficientnet.preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # Output layer
        outputs = self._build_output_layer(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        return model


class ResNet50Model(BaseModel):
    """ResNet50 with custom classification head."""

    def build(self) -> keras.Model:
        """Build ResNet50 architecture."""

        base_model = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )

        # Freeze base model
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Build custom head
        inputs = keras.Input(shape=self.input_shape)
        x = keras.applications.resnet50.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = self._build_output_layer(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        return model


class MobileNetV3Model(BaseModel):
    """MobileNetV3 with custom classification head."""

    def build(self) -> keras.Model:
        """Build MobileNetV3 architecture."""

        base_model = keras.applications.MobileNetV3Large(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )

        base_model.trainable = True
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = keras.applications.mobilenet_v3.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        outputs = self._build_output_layer(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        return model


class UNetModel(BaseModel):
    """U-Net architecture for medical image segmentation/classification."""

    def build(self) -> keras.Model:
        """Build U-Net architecture."""

        inputs = keras.Input(shape=self.input_shape)

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
        c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)

        # Decoder
        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.Concatenate()([u5, c3])
        c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u5)
        c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.Concatenate()([u6, c2])
        c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
        c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.Concatenate()([u7, c1])
        c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
        c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

        # Classification head
        gap = layers.GlobalAveragePooling2D()(c7)
        dense = layers.Dense(128, activation="relu")(gap)
        dense = layers.Dropout(0.5)(dense)

        outputs = self._build_output_layer(dense)

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        return model


def get_model(
    model_name: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
) -> BaseModel:
    """Factory function to get model by name."""

    models = {
        "cnn": CNNModel,
        "efficientnet_b3": EfficientNetB3Model,
        "resnet50": ResNet50Model,
        "mobilenet_v3": MobileNetV3Model,
        "unet": UNetModel,
    }

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return models[model_name](
        input_shape=input_shape, num_classes=num_classes, name=model_name
    )
