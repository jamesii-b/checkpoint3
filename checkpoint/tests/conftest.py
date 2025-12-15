
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def snapshot_dir(project_root):
    return project_root / "snapshots"

@pytest.fixture(scope="session")
def test_snapshot_path(snapshot_dir):
    return snapshot_dir / "test_checkpoint.bin"

@pytest.fixture(autouse=True)
def cleanup_test_snapshots(test_snapshot_path):
    yield
    if test_snapshot_path.exists():
        test_snapshot_path.unlink()
