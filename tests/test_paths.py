from pathlib import Path

from src.utils.paths import create_run_dir


def test_create_run_dir_creates_timestamped_directory(tmp_path: Path):
    run_dir = create_run_dir(base_dir=tmp_path, prefix="detect")

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
