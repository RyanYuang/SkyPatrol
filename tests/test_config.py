from pathlib import Path

import pytest

from src.utils.config import load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: demo\nvalue: 1\n", encoding="utf-8")

    data = load_yaml(config_path)

    assert data == {"name": "demo", "value": 1}


def test_load_yaml_missing_file_raises(tmp_path: Path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml(missing_path)
