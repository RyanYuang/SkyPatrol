from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FramePrediction:
    source: str
    frame_index: int
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)
    inference_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["detections"] = [d.to_dict() for d in self.detections]
        return payload
