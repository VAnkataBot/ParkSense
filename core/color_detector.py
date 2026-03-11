"""Color-based sign detector for Swedish parking signs.

Used as a fallback when Grounding DINO finds no detections, and as the
primary detector in the future Android port (same logic → Kotlin + OpenCV).

Detects signs by their distinctive colours:
  Blue   → parking / vehicle-restriction signs (H 95-135)
  Red    → no-parking / no-stopping signs      (H 0-10 and 165-180)
  Yellow → loading zone (lastplats) signs      (H 20-35)

No ML, no internet, works entirely on-device.
"""

import cv2
import numpy as np

# ── HSV color ranges ───────────────────────────────────────────────────────────

_BLUE_LO = np.array([95,  80,  40])
_BLUE_HI = np.array([135, 255, 255])

_RED_LO1 = np.array([0,   100, 80])
_RED_HI1 = np.array([10,  255, 255])
_RED_LO2 = np.array([165, 100, 80])
_RED_HI2 = np.array([180, 255, 255])

_YEL_LO  = np.array([20, 120, 120])
_YEL_HI  = np.array([35, 255, 255])

# ── Shape filters ──────────────────────────────────────────────────────────────

MIN_AREA_FRAC = 0.002   # min contour area as fraction of ROI area
MAX_ASPECT    = 3.0     # max(w,h) / min(w,h) — signs are roughly square
MIN_SOLIDITY  = 0.55    # contour area / convex hull area — rejects L-shapes etc.

# ── ROI — must match detector.py ──────────────────────────────────────────────

ROI_X_START = 0.28
ROI_X_END   = 0.72
ROI_Y_START = 0.00
ROI_Y_END   = 0.72


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_roi(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    x1 = int(W * ROI_X_START)
    x2 = int(W * ROI_X_END)
    y1 = int(H * ROI_Y_START)
    y2 = int(H * ROI_Y_END)
    return img[y1:y2, x1:x2], x1, y1


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    if color == "blue":
        return cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    if color == "red":
        return cv2.bitwise_or(
            cv2.inRange(hsv, _RED_LO1, _RED_HI1),
            cv2.inRange(hsv, _RED_LO2, _RED_HI2),
        )
    if color == "yellow":
        return cv2.inRange(hsv, _YEL_LO, _YEL_HI)
    return np.zeros(hsv.shape[:2], dtype=np.uint8)


def _boxes_from_mask(
    mask: np.ndarray, min_area: float
) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if area / (hull_area + 1e-6) < MIN_SOLIDITY:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]], iou_thr: float = 0.1
) -> list[tuple[int, int, int, int]]:
    """Merge boxes that overlap (same physical sign, multiple blobs)."""
    if not boxes:
        return []
    result = list(boxes)
    changed = True
    while changed:
        changed = False
        merged, used = [], [False] * len(result)
        for i in range(len(result)):
            if used[i]:
                continue
            x1, y1, x2, y2 = result[i]
            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = result[j]
                ix1, iy1 = max(x1, bx1), max(y1, by1)
                ix2, iy2 = min(x2, bx2), min(y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (bx2-bx1)*(by2-by1) - inter
                if union > 0 and inter / union > iou_thr:
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True
            merged.append((x1, y1, x2, y2))
            used[i] = True
        result = merged
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def detect_signs_by_color(
    img_array: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Detect parking signs by HSV color segmentation.

    Args:
        img_array: RGB uint8 full image.

    Returns:
        boxes:  (x1,y1,x2,y2) in full-image coords, sorted bottom-first.
        labels: DINO-style label string per box (used by classifier.py).
    """
    roi, ox, oy = _apply_roi(img_array)
    roi_h, roi_w = roi.shape[:2]
    min_area = roi_h * roi_w * MIN_AREA_FRAC

    hsv = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

    all_boxes: list[tuple[int, int, int, int]] = []
    all_labels: list[str] = []

    for color, label in [
        ("blue",   "blue parking sign"),
        ("red",    "no parking sign"),
        ("yellow", "lastplats loading zone sign"),
    ]:
        mask  = _color_mask(hsv, color)
        boxes = _boxes_from_mask(mask, min_area)
        boxes = _merge_overlapping(boxes)
        for b in boxes:
            all_boxes.append((b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy))
            all_labels.append(label)
            print(f"[color_detector] {color} at {b}")

    # Sort bottom-first (descending y2) — matches main.py's sort order
    paired = sorted(zip(all_boxes, all_labels), key=lambda p: p[0][3], reverse=True)
    if paired:
        boxes_out, labels_out = zip(*paired)
        return list(boxes_out), list(labels_out)
    return [], []
