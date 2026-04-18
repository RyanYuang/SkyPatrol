from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.visdrone import build_train_kwargs, default_visdrone_root, write_visdrone_runtime_yaml
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on VisDrone2019-DET")
    parser.add_argument("--config", default="configs/model/yolo_visdrone_train.yaml", help="training config yaml")
    parser.add_argument("--dataset-root", default=None, help="where VisDrone will be stored locally")
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--download-only", action="store_true", help="only download/prepare dataset, do not train")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    args = parse_args()
    train_config = load_yaml(PROJECT_ROOT / args.config)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else default_visdrone_root()
    runtime_yaml_path = write_visdrone_runtime_yaml(dataset_root=dataset_root)

    dataset_info = check_det_dataset(str(runtime_yaml_path), autodownload=True)
    print(f"[OK] VisDrone dataset ready: {dataset_info['path']}")
    print(f"[INFO] train: {dataset_info['train']}")
    print(f"[INFO] val:   {dataset_info['val']}")

    if args.download_only:
        print("[DONE] Download-only mode completed.")
        return

    train_kwargs = build_train_kwargs(
        train_config=train_config,
        data_yaml_path=runtime_yaml_path,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    model = YOLO(train_kwargs.pop("model"))
    print(f"[INFO] Training kwargs: {train_kwargs}")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
