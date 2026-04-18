from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISDRONE_CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
VISDRONE_DOWNLOAD_SCRIPT = """import shutil
from pathlib import Path

from ultralytics.utils import ASSETS_URL, TQDM
from ultralytics.utils.downloads import download


def visdrone2yolo(dir, split, source_name=None):
    from PIL import Image

    source_dir = dir / (source_name or f\"VisDrone2019-DET-{split}\")
    images_dir = dir / \"images\" / split
    labels_dir = dir / \"labels\" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    if (source_images_dir := source_dir / \"images\").exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in source_images_dir.glob(\"*.jpg\"):
            img.rename(images_dir / img.name)

    for f in TQDM((source_dir / \"annotations\").glob(\"*.txt\"), desc=f\"Converting {split}\"):
        img_size = Image.open(images_dir / f.with_suffix(\".jpg\").name).size
        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines = []

        with open(f, encoding=\"utf-8\") as file:
            for row in [x.split(\",\") for x in file.read().strip().splitlines()]:
                if row[4] != \"0\":
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1
                    x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                    w_norm, h_norm = w * dw, h * dh
                    lines.append(f\"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\\n\")

        (labels_dir / f.name).write_text(\"\".join(lines), encoding=\"utf-8\")


dir = Path(yaml[\"path\"])
urls = [
    f\"{ASSETS_URL}/VisDrone2019-DET-train.zip\",
    f\"{ASSETS_URL}/VisDrone2019-DET-val.zip\",
    f\"{ASSETS_URL}/VisDrone2019-DET-test-dev.zip\",
]
download(urls, dir=dir, threads=4)

splits = {\"VisDrone2019-DET-train\": \"train\", \"VisDrone2019-DET-val\": \"val\", \"VisDrone2019-DET-test-dev\": \"test\"}
for folder, split in splits.items():
    visdrone2yolo(dir, split, folder)
    shutil.rmtree(dir / folder)
"""


def default_visdrone_root() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "VisDrone"


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else prefix for line in text.splitlines())


def write_visdrone_runtime_yaml(dataset_root: str | Path | None = None, output_path: str | Path | None = None) -> Path:
    dataset_root_path = Path(dataset_root) if dataset_root else default_visdrone_root()
    dataset_root_path = dataset_root_path.resolve()
    runtime_yaml_path = Path(output_path) if output_path else PROJECT_ROOT / "data" / "interim" / "visdrone_detection_runtime.yaml"
    runtime_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names_yaml = yaml.safe_dump(VISDRONE_CLASS_NAMES, sort_keys=False, allow_unicode=True).strip()
    content = (
        f"path: {dataset_root_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{_indent_block(names_yaml, 2)}\n"
        "download: |\n"
        f"{_indent_block(VISDRONE_DOWNLOAD_SCRIPT.rstrip(), 2)}\n"
    )
    runtime_yaml_path.write_text(content, encoding="utf-8")
    return runtime_yaml_path


def build_train_kwargs(
    train_config: Dict[str, Any],
    data_yaml_path: str | Path,
    device: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
) -> Dict[str, Any]:
    return {
        "model": train_config.get("model_name", "yolov8n.pt"),
        "data": str(Path(data_yaml_path).resolve()),
        "epochs": epochs if epochs is not None else int(train_config.get("epochs", 100)),
        "imgsz": imgsz if imgsz is not None else int(train_config.get("imgsz", 960)),
        "batch": batch if batch is not None else int(train_config.get("batch", 8)),
        "workers": int(train_config.get("workers", 4)),
        "device": device or train_config.get("device", "cpu"),
        "project": train_config.get("project", "runs/train"),
        "name": train_config.get("name", "visdrone-yolov8n"),
        "patience": int(train_config.get("patience", 20)),
        "cache": bool(train_config.get("cache", False)),
        "optimizer": train_config.get("optimizer", "auto"),
        "close_mosaic": int(train_config.get("close_mosaic", 10)),
        "seed": int(train_config.get("seed", 42)),
        "deterministic": bool(train_config.get("deterministic", True)),
        "pretrained": bool(train_config.get("pretrained", True)),
        "exist_ok": bool(train_config.get("exist_ok", True)),
        "single_cls": bool(train_config.get("single_cls", False)),
        "degrees": float(train_config.get("degrees", 0.0)),
        "translate": float(train_config.get("translate", 0.1)),
        "scale": float(train_config.get("scale", 0.5)),
        "fliplr": float(train_config.get("fliplr", 0.5)),
        "mosaic": float(train_config.get("mosaic", 1.0)),
        "mixup": float(train_config.get("mixup", 0.0)),
        "copy_paste": float(train_config.get("copy_paste", 0.0)),
    }
