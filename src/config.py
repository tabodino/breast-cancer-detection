"""Application configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.utils.mlflow_path import get_crossplatform_mlflow_uri

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_format_console: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    log_format_file: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    log_rotation: str = "10 MB"
    log_retention: str = "1 week"
    log_compression: str = "zip"

    # Paths
    projet_root: Path = Path(__file__).parent.parent.resolve()
    data_dir: Path = Path("data")
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")
    artifacts_dir: Path = Path("artifacts")

    # MLflow Configuration
    mlflow_tracking_uri: str = get_crossplatform_mlflow_uri()
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "breast-cancer-detection"
    )
    mlflow_registry_uri: str = get_crossplatform_mlflow_uri()

    # Dataset url
    dataset_url: str = (
        "https://data.mendeley.com/public-files/datasets/k4t7msnt3y/files/"
        "f0bf474c-91cf-4db7-85b0-2273569a7b59/file_downloaded"
    )

    # Image Processing
    iamge_size: tuple = (224, 224)
    image_channels: int = 3

    # Model Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Random seed for reproducibility
    random_seed: int = 42

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()
