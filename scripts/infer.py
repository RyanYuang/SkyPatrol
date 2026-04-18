from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.detector import YoloDetector
from src.schemas.prediction import FramePrediction
from src.utils.config import load_yaml
from src.utils.io import (
    create_video_writer,
    is_camera_source,
    is_image_file,
    is_video_file,
    save_image,
    save_predictions_json,
)
from src.utils.paths import create_run_dir
from src.utils.visualizer import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV Vision MVP0 inference entrypoint")
    parser.add_argument("--source", required=True, help="Image path, video path, or camera index like 0")
    parser.add_argument("--model-config", default="configs/model/yolo_detect.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/infer.yaml")
    parser.add_argument("--class-config", default="configs/data/class_names.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    parser.add_argument("--show", action="store_true", help="Force display window")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    return parser.parse_args()


def resolve_show_flag(cli_args: argparse.Namespace, runtime_config: dict) -> bool:
    if cli_args.show:
        return True
    if cli_args.no_show:
        return False
    return bool(runtime_config.get("show", True))


def load_detector(args: argparse.Namespace) -> tuple[YoloDetector, dict]:
    model_config = load_yaml(PROJECT_ROOT / args.model_config)
    runtime_config = load_yaml(PROJECT_ROOT / args.runtime_config)
    class_config = load_yaml(PROJECT_ROOT / args.class_config)
    class_names = {int(k): v for k, v in class_config.get("names", {}).items()}
    detector = YoloDetector(model_config=model_config, class_names=class_names, device=args.device)
    return detector, runtime_config


def run_on_image(source: str, detector: YoloDetector, runtime_config: dict, output_dir: Path) -> None:
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {source}")

    prediction = detector.predict_frame(image, source_name=source, frame_index=0)
    annotated = draw_detections(
        image,
        prediction.detections,
        line_width=int(runtime_config.get("line_width", 2)),
        font_scale=float(runtime_config.get("font_scale", 0.5)),
    )

    if runtime_config.get("save_image", True):
        image_name = f"annotated_{Path(source).name}"
        save_image(output_dir / image_name, annotated)

    if runtime_config.get("save_json", True):
        save_predictions_json(output_dir / "predictions.json", [prediction])

    print(f"[DONE] image inference completed: {output_dir}")


def run_on_video(source: str, detector: YoloDetector, runtime_config: dict, output_dir: Path, show: bool) -> None:
    capture = cv2.VideoCapture(int(source) if is_camera_source(source) else source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    predictions: List[FramePrediction] = []
    frame_index = 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or float(runtime_config.get("camera_output_fps", 20.0))
    if fps <= 1e-6:
        fps = float(runtime_config.get("camera_output_fps", 20.0))

    writer = None
    writer_path = None
    if runtime_config.get("save_video", True):
        output_name = "annotated_camera.mp4" if is_camera_source(source) else f"annotated_{Path(source).stem}.mp4"
        writer, writer_path = create_video_writer(output_dir / output_name, width, height, fps)

    max_frames = runtime_config.get("max_frames")
    last_log_time = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            prediction = detector.predict_frame(frame, source_name=source, frame_index=frame_index)
            predictions.append(prediction)
            annotated = draw_detections(
                frame,
                prediction.detections,
                line_width=int(runtime_config.get("line_width", 2)),
                font_scale=float(runtime_config.get("font_scale", 0.5)),
            )

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow("UAV Vision MVP0", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] ESC pressed, stopping inference.")
                    break

            if time.time() - last_log_time >= 1.0:
                current_fps = len(predictions) / max(1e-6, sum((p.inference_ms or 0.0) for p in predictions) / 1000.0)
                print(f"[INFO] frames={len(predictions)} avg_model_fps={current_fps:.2f}")
                last_log_time = time.time()

            frame_index += 1
            if max_frames is not None and frame_index >= int(max_frames):
                print(f"[INFO] reached max_frames={max_frames}, stopping.")
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if runtime_config.get("save_json", True):
        save_predictions_json(output_dir / "predictions.json", predictions)

    if writer_path:
        print(f"[DONE] video saved to: {writer_path}")
    print(f"[DONE] predictions saved to: {output_dir / 'predictions.json'}")


def main() -> None:
    args = parse_args()
    detector, runtime_config = load_detector(args)
    show = resolve_show_flag(args, runtime_config)
    output_dir = Path(args.output_dir) if args.output_dir else create_run_dir(prefix="detect")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_image_file(args.source):
        run_on_image(args.source, detector, runtime_config, output_dir)
        return

    if is_video_file(args.source) or is_camera_source(args.source):
        run_on_video(args.source, detector, runtime_config, output_dir, show)
        return

    raise ValueError(f"Unsupported source type: {args.source}")


if __name__ == "__main__":
    main()
