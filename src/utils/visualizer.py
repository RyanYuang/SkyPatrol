from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from src.schemas.prediction import Detection


_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 0, 0)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    line_width: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _COLOR, line_width)
        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(annotated, (x1, top), (x1 + text_w + 6, top + text_h + baseline + 4), _COLOR, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return annotated
