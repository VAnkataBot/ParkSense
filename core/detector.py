from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLOE
from ultralytics.utils import SETTINGS

SETTINGS["weights_dir"] = str(Path(__file__).parent.parent / "models")

MODEL_PATH = "models/yoloe-26s-seg.pt"
CLASS_NAMES = [
    # Main signs
    "blue square swedish parking allowed sign with white letter P",
    "round red circle swedish no parking prohibited sign",
    "round red circle swedish area-wide parking prohibition zone sign",
    "round red circle with diagonal cross swedish end of parking prohibition zone sign",
    # Additional plates
    "rectangular white plate with hours and days text swedish time restricted parking sign",
    "rectangular white plate with coin or fee text swedish paid parking sign",
    "rectangular white plate with permit text swedish special permit required parking sign",
    "rectangular white plate with visitor or customer text swedish visitors only parking sign",
    "rectangular white plate with parking disc clock swedish parking disc required sign",
    "rectangular white plate with tenant or boende text swedish residents only parking sign",
    # Symbol plates
    "square blue plate with international wheelchair accessibility symbol white icon",
    "rectangular white plate with heavy truck lorry vehicle silhouette icon",
    "rectangular white plate with passenger car sedan vehicle silhouette icon",
    "rectangular white plate with motorcycle moped vehicle silhouette icon",
    "rectangular white plate with trailer caravan vehicle silhouette icon",
    "rectangular white plate with electric vehicle charging plug lightning bolt icon",
    # Boundary plates
    "rectangular white plate with directional arrow indicating parking zone boundary",
]
CONF_THRESHOLD = 0.5

_model = None
_model_classes = None


def _get_model() -> YOLOE:
    global _model, _model_classes
    if _model is None or _model_classes != CLASS_NAMES:
        _model = YOLOE(MODEL_PATH)
        _model.set_classes(CLASS_NAMES, _model.get_text_pe(CLASS_NAMES))
        _model_classes = CLASS_NAMES[:]
    return _model



def detect_sign(img_array: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]], List[str]]:
    """
    Detect parking signs using YOLOe.

    Returns:
        Tuple of (combined crop or None, list of bounding boxes, list of detected class names).
    """
    if img_array is None or img_array.size == 0:
        return None, [], []

    model = _get_model()
    results = model.predict(img_array, conf=CONF_THRESHOLD, verbose=False)

    boxes = []
    detected_classes = []
    for result in results:
        for box in result.boxes or []:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            print(f"[detector] detected box=({x1},{y1},{x2},{y2}) conf={conf:.4f} class={cls_name}")
            boxes.append((x1, y1, x2, y2))
            detected_classes.append(cls_name)

    if not boxes:
        return None, [], []

    # OCR the full vertical span of all detections so panels between boxes
    # (e.g. "0–13 m" duration plate) are not missed when YOLO skips them.
    h, w = img_array.shape[:2]
    span_x1 = max(0, min(b[0] for b in boxes))
    span_y1 = max(0, min(b[1] for b in boxes))
    span_x2 = min(w, max(b[2] for b in boxes))
    span_y2 = min(h, max(b[3] for b in boxes))
    combined = img_array[span_y1:span_y2, span_x1:span_x2]

    if combined.size == 0:
        return None, boxes, detected_classes

    return combined, boxes, detected_classes


def draw_boxes(img_array: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw bounding boxes on a copy of the image."""
    annotated_bgr = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color=(0, 200, 0), thickness=3)
        cv2.putText(
            annotated_bgr,
            f"Sign {i + 1}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 0),
            2,
        )

    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
