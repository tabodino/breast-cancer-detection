from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
import mlflow
from loguru import logger
from src.config import get_settings
from src.models.base_model import get_model


settings = get_settings()


class ModelTrainer:
    """
    Handles model training with MLflow experiment tracking.
    Manages hyperparameters, callbacks, and artifact logging.
    """

    def __init__(self, model_name: str = "efficientnet_b3"):
        self.model_name = model_name
        self.model = None
        self.history = None

        # Setup MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        # Set random seeds for reproducibility
        np.random.seed(settings.random_seed)
        tf.random.set_seed(settings.random_seed)

    def build_model(self):
        """Build the model architecture."""
        self.model = get_model(self.model_name)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(),
            ],
        )
        logger.info(f"Model {self.model_name} built successfully")

    def get_callbacks(self, run_id: str) -> list:
        """Create training callbacks."""

        model_checkpoint = ModelCheckpoint(
            settings.models_dir / f"best_model_{self.model_name}_{run_id}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=settings.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )

        return [model_checkpoint, early_stopping, reduce_lr]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict = None,
    ) -> Dict:
        """
        Train the model with MLflow tracking.
        """

        if hyperparams is None:
            hyperparams = {
                "learning_rate": settings.learning_rate,
                "batch_size": settings.batch_size,
                "epochs": settings.epochs,
                "dropout": 0.3,
            }

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            logger.info(f"Starting training run: {run_id}")

            # Build model
            self.build_model()

            # Log hyperparameters
            mlflow.log_params({"model_architecture": self.model_name, **hyperparams})

            # Train model
            callbacks = self.get_callbacks(run_id)

            self.history = self.model.get_model().fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=hyperparams.get("epochs", settings.epochs),
                batch_size=hyperparams.get("batch_size", settings.batch_size),
                callbacks=callbacks,
                verbose=1,
            )

            # Evaluate on validation set
            val_loss, val_acc, val_precision, val_recall, val_auc = (
                self.model.get_model().evaluate(X_val, y_val, verbose=0)
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "val_precision": float(val_precision),
                    "val_recall": float(val_recall),
                    "val_auc": float(val_auc),
                }
            )

            # Save model
            model_filename = f"model_{self.model_name}_{run_id}.keras"
            model_path = settings.models_dir / model_filename
            self.model.save(model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")

            logger.info(f"Training completed. Model saved to {model_path}")

            return {
                "run_id": run_id,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_auc": float(val_auc),
            }


def train_multiple_models(
    model_names: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict:
    """
    Train multiple models and compare results.
    """

    results = {}

    for model_name in model_names:
        logger.info(f"Training {model_name}...")

        trainer = ModelTrainer(model_name=model_name)
        result = trainer.train(X_train, y_train, X_val, y_val)
        results[model_name] = result

        logger.info(f"{model_name} results: {result}")

    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Training script ready. Import and use ModelTrainer class.")
