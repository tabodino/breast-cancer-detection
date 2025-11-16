import os
from pathlib import Path
from typing import Optional, Tuple


def get_crossplatform_mlflow_uri(folder: str = "mlruns") -> str:
    """
    Returns an OS/MLflow-safe absolute file:// URI to an mlruns dir (creates it if not exists).
    """
    base_dir: Path = Path(__file__).resolve().parents[2]
    mlruns_path: Path = (base_dir / folder).resolve()
    mlruns_path.mkdir(exist_ok=True)
    uri: str = f"file:///{str(mlruns_path).replace(os.sep, '/')}"
    return uri


def get_latest_run_id(models_dir: Path = Path("models")) -> Optional[Tuple[str, str]]:
    """Get latest saved model Path and run_id."""
    files = sorted(
        models_dir.glob("best_model_*.keras"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if files:
        file = files[0]
        run_id = file.stem.replace("best_model_", "")
        return run_id
    return None
