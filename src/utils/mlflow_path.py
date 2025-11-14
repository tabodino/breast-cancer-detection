from pathlib import Path
import os


def get_crossplatform_mlflow_uri(folder: str = "mlruns") -> str:
    """
    Returns an OS/MLflow-safe absolute file:// URI to an mlruns dir (creates it if not exists).
    """
    base_dir: Path = Path(__file__).resolve().parents[2]
    mlruns_path: Path = (base_dir / folder).resolve()
    mlruns_path.mkdir(exist_ok=True)
    uri: str = f"file:///{str(mlruns_path).replace(os.sep, '/')}"
    return uri
