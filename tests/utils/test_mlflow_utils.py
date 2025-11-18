import time
from src.utils import get_latest_run_id


def test_get_latest_run_id(tmp_path):
    models_dir = tmp_path

    assert get_latest_run_id(models_dir) is None

    run_ids = ["abc123", "def456"]
    fp1 = models_dir / f"best_model_{run_ids[0]}.keras"
    fp2 = models_dir / f"best_model_{run_ids[1]}.keras"

    fp1.touch()
    time.sleep(0.1)
    fp2.touch()

    old_time = time.time() - 100
    fp1.stat()
    fp1.utime((old_time, old_time))
    fp2.utime((time.time(), time.time()))

    assert get_latest_run_id(models_dir) == run_ids[1]


def test_get_latest_run_id_single(tmp_path):
    models_dir = tmp_path
    run_id = "xyz789"
    fp = models_dir / f"best_model_{run_id}.keras"
    fp.touch()
    assert get_latest_run_id(models_dir) == run_id


def test_get_latest_run_id_other_files(tmp_path):
    models_dir = tmp_path
    (models_dir / "not_a_model.txt").touch()
    (models_dir / "best_model_.txt").touch()
    assert get_latest_run_id(models_dir) is None
