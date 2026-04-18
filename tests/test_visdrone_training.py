from pathlib import Path

from src.training.visdrone import build_train_kwargs, write_visdrone_runtime_yaml
from src.utils.config import load_yaml


def test_write_visdrone_runtime_yaml_uses_absolute_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "VisDrone"
    runtime_yaml = tmp_path / "visdrone_runtime.yaml"

    output_path = write_visdrone_runtime_yaml(dataset_root=dataset_root, output_path=runtime_yaml)
    data = load_yaml(output_path)

    assert output_path == runtime_yaml
    assert data["path"] == str(dataset_root.resolve())
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert "download" in data
    assert "VisDrone2019-DET-train.zip" in data["download"]


def test_build_train_kwargs_applies_cli_overrides(tmp_path: Path):
    data_yaml = tmp_path / "visdrone_runtime.yaml"
    data_yaml.write_text("path: /tmp/demo\ntrain: images/train\nval: images/val\nnames: {0: car}\n", encoding="utf-8")

    config = {
        "model_name": "yolov8n.pt",
        "imgsz": 960,
        "epochs": 100,
        "batch": 8,
        "workers": 4,
        "device": "cpu",
        "project": "runs/train",
        "name": "visdrone-yolov8n",
        "patience": 20,
        "cache": False,
        "optimizer": "auto",
        "close_mosaic": 10,
        "seed": 42,
        "deterministic": True,
        "pretrained": True,
        "exist_ok": True,
        "single_cls": False,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    kwargs = build_train_kwargs(
        train_config=config,
        data_yaml_path=data_yaml,
        device="mps",
        epochs=5,
        imgsz=640,
        batch=2,
    )

    assert kwargs["data"] == str(data_yaml.resolve())
    assert kwargs["model"] == "yolov8n.pt"
    assert kwargs["device"] == "mps"
    assert kwargs["epochs"] == 5
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 2
    assert kwargs["project"] == "runs/train"
    assert kwargs["name"] == "visdrone-yolov8n"
