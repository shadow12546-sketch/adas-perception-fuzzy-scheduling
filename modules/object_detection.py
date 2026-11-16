import numpy as np
from ultralytics import YOLO

from config import MODEL_SIZE, CONFIDENCE_THRESHOLD


COCO_ADAS_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

# Maps your fine-tuned dataset labels to the canonical labels
# used by the rest of the ADAS pipeline.
CUSTOM_CLASS_MAP = {
    "biker": (1, "bicycle"),
    "car": (2, "car"),
    "pedestrian": (0, "person"),
    "trafficlight": (9, "traffic light"),
    "trafficlight-green": (9, "traffic light"),
    "trafficlight-greenleft": (9, "traffic light"),
    "trafficlight-red": (9, "traffic light"),
    "trafficlight-redleft": (9, "traffic light"),
    "trafficlight-yellow": (9, "traffic light"),
    "trafficlight-yellowleft": (9, "traffic light"),
    "truck": (7, "truck"),
}

CUSTOM_MODEL_MARKERS = {"biker", "pedestrian", "trafficlight"}


class ObjectDetector:
    def __init__(
        self,
        model_size: str = MODEL_SIZE,
        confidence: float = CONFIDENCE_THRESHOLD,
    ):
        self.model = YOLO(model_size)
        self.confidence = confidence
        self.relevant_classes = dict(COCO_ADAS_CLASSES)

        model_names = self.model.names
        if isinstance(model_names, dict):
            self.model_names = {
                int(class_id): str(name)
                for class_id, name in model_names.items()
            }
        else:
            self.model_names = {
                class_id: str(name)
                for class_id, name in enumerate(model_names)
            }

        normalized_names = {
            name.strip().lower() for name in self.model_names.values()
        }
        self.uses_custom_labels = bool(
            normalized_names & CUSTOM_MODEL_MARKERS
        )

        model_type = (
            "custom-class"
            if self.uses_custom_labels
            else "COCO-class"
        )
        print(f"Object detector loaded: {model_size} ({model_type} mapping)")

    def detect(self, frame: np.ndarray) -> list:
        """Return ADAS detections with canonical class IDs and labels."""
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.confidence,
        )
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                model_class_id = int(box.cls[0])
                model_class_name = self.model_names.get(
                    model_class_id,
                    str(model_class_id),
                )

                if self.uses_custom_labels:
                    mapped_class = CUSTOM_CLASS_MAP.get(
                        model_class_name.strip().lower()
                    )
                    if mapped_class is None:
                        continue
                    class_id, class_name = mapped_class
                else:
                    if model_class_id not in COCO_ADAS_CLASSES:
                        continue
                    class_id = model_class_id
                    class_name = COCO_ADAS_CLASSES[model_class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": class_id,
                    "class_name": class_name,
                    "model_class_id": model_class_id,
                    "model_class_name": model_class_name,
                })

        return detections
