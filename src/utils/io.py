from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.schemas.prediction import FramePrediction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def is_image_file(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(source: str) -> bool:
    return Path(source).suffix.lower() in VIDEO_EXTENSIONS


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output_path = ensure_parent(path)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save image to {output_path}")
    return output_path


def create_video_writer(path: str | Path, width: int, height: int, fps: float):
    output_path = ensure_parent(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer, output_path


def save_predictions_json(path: str | Path, predictions: Iterable[FramePrediction]) -> Path:
    output_path = ensure_parent(path)
    payload = [prediction.to_dict() for prediction in predictions]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
