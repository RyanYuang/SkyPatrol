from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(base_dir: str | Path | None = None, prefix: str = "detect") -> Path:
    root = Path(base_dir) if base_dir else PROJECT_ROOT / "runs" / prefix
    run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
