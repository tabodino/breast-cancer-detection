import pytest
import types
import zipfile
from unittest.mock import Mock, patch
from src.etl.download_dataset import (
    download_file, extract_zip, cleanup_raw_folder
)

@pytest.fixture
def temp_dir(tmp_path):
    # Temporary directory for raw_data
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_download_file(monkeypatch, temp_dir):
    dest = temp_dir / "test.zip"
    url = "http://test.local/fakezip"

    class FakeResponse:
        def __init__(self):
            self.headers = {'content-length': '6'}
            self._data = [b'abc', b'def']
        def raise_for_status(self):
            pass


        def iter_content(self, size):
            return self._data
    # Monkeypatch requests.get
    monkeypatch.setattr("requests.get", lambda *a, **kw: FakeResponse())
    # Monkeypatch tqdm to identity contextmanager
    monkeypatch.setattr("tqdm.tqdm", lambda *a, **kw: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, e, f, g: None, update=lambda x: None))
    
    # The loguru logger is already used globally; outputs to stdout, but does not interfere
    download_file(url, dest)
    assert dest.exists()
    assert dest.read_bytes() == b'abcdef'


def test_extract_zip(temp_dir):
    # Create a zipfile with a single file inside
    zip_path = temp_dir / "some.zip"
    extract_to = temp_dir / "extracted"
    extract_to.mkdir()
    inside_file = "data.txt"
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr(inside_file, "hello world")
    extract_zip(zip_path, extract_to)
    assert (extract_to / inside_file).exists()
    assert (extract_to / inside_file).read_text() == "hello world"


def test_cleanup_raw_folder_with_mocks():
    fake_path = Mock()
    fake_zip = Mock()
  
    fake_zip.exists.return_value = True

    fake_subdir = Mock()
    fake_subdir.is_dir.return_value = True
    fake_subdir.iterdir.return_value = []
    fake_subdir.exists.return_value = True

    fake_path.iterdir.return_value = [fake_subdir]
    fake_path.exists.return_value = True

    # Monkeypatch settings
    with patch("src.etl.download_dataset.settings") as settings_mock:
        settings_mock.raw_data_dir = fake_path
        cleanup_raw_folder(fake_zip)

    fake_subdir.rmdir.assert_called_once()
    fake_zip.unlink.assert_called_once()
