from __future__ import annotations

from pathlib import Path

DIRECTORIES = [
    "configs/model",
    "configs/data",
    "configs/runtime",
    "assets/demo",
    "checkpoints",
    "data/raw",
    "data/interim",
    "data/processed",
    "runs",
    "scripts",
    "src/pipeline",
    "src/utils",
    "src/schemas",
    "tests",
    "docs",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative_dir in DIRECTORIES:
        directory = project_root / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {directory}")


if __name__ == "__main__":
    main()
