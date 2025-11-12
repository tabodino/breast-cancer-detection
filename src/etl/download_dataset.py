import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
from src.config import get_settings
from loguru import logger

settings = get_settings()


def download_file(url: str, dest: Path):
    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    file_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
    ):
        for chunk in response.iter_content(1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))
    logger.success(f"Downloaded to {dest}")


def extract_zip(zip_path: Path, extract_to: Path):
    logger.info(f"Extracting archive to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.success("Extraction complete.")


def cleanup_raw_folder(zip_path: str):
    """Clean the raw data directory after downloading a new dataset."""
    if settings.raw_data_dir.exists():
        if zip_path.exists():
            zip_path.unlink()
            logger.info(f"Deleted archive: {zip_path}")

        # Remove any empty subdirectories in raw_dir
        for subdir in settings.raw_data_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
                logger.info(f"Removed empty folder: {subdir}")

        logger.success("Cleanup of raw directory complete.")


if __name__ == "__main__":  # pragma: no cover
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = settings.raw_data_dir / "mendeley.zip"
    download_file(settings.dataset_url, zip_path)
    extract_zip(zip_path, settings.raw_data_dir)
    cleanup_raw_folder(zip_path)
