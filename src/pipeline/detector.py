from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.schemas.prediction import Detection, FramePrediction


class YoloDetector:
    def __init__(self, model_config: Dict[str, Any], class_names: Dict[int, str] | None = None, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_config.get("model_name", "yolov8n.pt")
        self.imgsz = int(model_config.get("imgsz", 640))
        self.conf = float(model_config.get("conf", 0.25))
        self.iou = float(model_config.get("iou", 0.45))
        self.device = device or model_config.get("device", "cpu")
        self.classes = model_config.get("classes")
        self.verbose = bool(model_config.get("verbose", False))
        self.class_names = class_names or {}
        self.model = YOLO(self.model_name)

    def predict_frame(self, frame: np.ndarray, source_name: str, frame_index: int = 0) -> FramePrediction:
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=self.verbose,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]
        boxes = result.boxes
        detections: List[Detection] = []

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            for bbox, confidence, class_id_raw in zip(xyxy, confs, class_ids):
                class_id = int(class_id_raw)
                class_name = self._resolve_class_name(class_id, result)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox_xyxy=[float(v) for v in bbox],
                    )
                )

        height, width = frame.shape[:2]
        return FramePrediction(
            source=source_name,
            frame_index=frame_index,
            image_width=width,
            image_height=height,
            detections=detections,
            inference_ms=round(elapsed_ms, 3),
        )

    def _resolve_class_name(self, class_id: int, result: Any) -> str:
        if class_id in self.class_names:
            return self.class_names[class_id]
        names = getattr(result, "names", {}) or {}
        return str(names.get(class_id, class_id))
