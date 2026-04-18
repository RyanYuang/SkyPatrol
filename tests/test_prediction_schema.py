from src.schemas.prediction import Detection, FramePrediction


def test_frame_prediction_to_dict_serializes_nested_detections():
    frame_prediction = FramePrediction(
        source="demo.jpg",
        frame_index=0,
        image_width=640,
        image_height=480,
        detections=[
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.91,
                bbox_xyxy=[10.0, 20.0, 100.0, 160.0],
            )
        ],
        inference_ms=15.5,
    )

    payload = frame_prediction.to_dict()

    assert payload["source"] == "demo.jpg"
    assert payload["detections"][0]["class_name"] == "car"
    assert payload["detections"][0]["bbox_xyxy"] == [10.0, 20.0, 100.0, 160.0]
