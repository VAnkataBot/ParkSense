"""Sign detection using Grounding DINO zero-shot object detection.

Pre-processing
--------------
Portrait phone photo → ROI crop before DINO runs:
  • Horizontal: centre 40% of width  (was 60% — much tighter, kills side poles)
  • Vertical:   top 70% of height    (was 78% — cuts more ground noise)

Post-processing pipeline
------------------------
  1. Blocklist   — drop non-signs by label keyword
  2. NMS         — remove duplicates
  3. Anchor      — best main sign (largest × most centred in ROI)
  4. Pole band   — keep only boxes within anchor_width × POLE_BAND_TOLERANCE
                   of the anchor centre (not edge) — much tighter than before
  5. Size gate   — drop boxes below MIN_BOX_AREA_FRACTION of ROI area
"""

from typing import Optional
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"

# ── Labels ─────────────────────────────────────────────────────────────────────
LABELS = (
    "blue parking sign . "
    "no parking sign . "
    "no stopping sign . "
    "lastplats loading zone sign . "   # yellow sign with red X = no stopping/loading only
    "handicap parking sign . "
    "electric vehicle charging sign . "
    "truck parking sign . "
    "motorcycle parking sign . "
    "trailer parking sign . "
    "parking disc time limit sign . "
    "diagonal parking sign . "
    "parallel parking sign . "
    "residents parking sign . "
    "exception time plate . "
    "arrow direction plate . "
    "övrig tid parking sign"           # blue P with övrig tid text
)

BOX_THRESHOLD  = 0.22
TEXT_THRESHOLD = 0.18

# ── ROI crop ───────────────────────────────────────────────────────────────────
# Tighter horizontal band — pole is always near centre in portrait shots.
# If your pole is being clipped, widen by 0.05 on each side.
ROI_X_START = 0.28   # keep from 28% inward
ROI_X_END   = 0.72   # keep up to 72%  → only centre 44% of width
ROI_Y_START = 0.00   # signs can start at very top
ROI_Y_END   = 0.72   # was 0.72 — extended to catch blue P signs lower on pole

# ── Pole band ──────────────────────────────────────────────────────────────────
# Measured from ANCHOR CENTRE (not edge), in multiples of anchor half-width.
# 1.5 means a box centre must be within 1.5× the anchor's half-width of the
# anchor centre.  Smaller = stricter.  Was effectively ~2.8 before.
POLE_BAND_HALF_WIDTHS = 1.5

# Minimum box area as fraction of ROI area — blocks tiny distant signs
MIN_BOX_AREA_FRACTION = 0.003

# ── Blocklist ──────────────────────────────────────────────────────────────────
_BLOCKLIST = [
    "bicycle", "bike", "cyclist",
    "person", "pedestrian",
    "car", "vehicle", "bus",
    "tree", "building", "window",
]

# ── Label → class ──────────────────────────────────────────────────────────────
_LABEL_MAP = [
    # ── Specific prohibitions first (must beat generic "parking") ──────────
    ("no stopping",      "no_stopping"),
    ("no parking",       "no_parking"),
    ("loading zone",     "loading_zone"),
    ("lastplats",        "loading_zone"),
    ("loading",          "loading_zone"),
    ("övrig tid",        "parking"),       # blue P with övrig tid → parking anchor
    # ── Vehicle-specific anchors ───────────────────────────────────────────
    ("handicap",         "handicap"),
    ("wheelchair",       "handicap"),
    ("electric vehicle", "ev_charging"),
    ("charging",         "ev_charging"),
    ("motorcycle",       "motorcycle"),
    ("truck",            "truck"),
    ("diagonal",         "diagonal_parking"),   # before "trailer" — diagonal beats trailer on tie
    ("parallel",         "parallel_parking"),
    ("trailer parking sign", "trailer"),        # full phrase required
    # ── Modifier / sub-plates ──────────────────────────────────────────────
    # "exception time plate" and "parking disc time limit" must come BEFORE
    # "parking" so they don't fall through to the generic parking class.
    ("exception",        "exception_plate"),
    ("time limit",       "exception_plate"),   # was parking_disc — it's a time window plate
    ("parking disc",     "parking_disc"),       # the actual clock-symbol plate
    ("residents",        "residents"),
    ("arrow",            "arrow_plate"),
    ("distance",         "distance_plate"),
    # ── Generic parking anchor — keep last ────────────────────────────────
    ("parking",          "parking"),
]

_processor = None
_model     = None


# ── Model ──────────────────────────────────────────────────────────────────────

def _get_model():
    global _processor, _model
    if _model is None:
        print("[detector] Loading Grounding DINO…")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        _model.eval()
        if torch.cuda.is_available():
            _model.cuda()
            print("[detector] GPU ready.")
        else:
            print("[detector] CPU mode.")
    return _processor, _model


# ── Utilities ──────────────────────────────────────────────────────────────────

def _label_to_class(label: str) -> str:
    """
    DINO returns compound labels like
    'no parking sign no stopping sign motorcycle parking sign diagonal parking sign'.
    Vote for each class, boost specific classes so they beat the generic
    'parking' which appears in almost every compound label as noise.
    """
    ll = label.lower()
    votes: dict[str, int] = {}
    for kw, cls in _LABEL_MAP:
        if kw in ll:
            votes[cls] = votes.get(cls, 0) + 1

    if not votes:
        return "parking"

    # Prohibition/loading anchors get +3 — highest priority
    _PROHIBIT_BOOST = {"no_stopping", "no_parking", "loading_zone"}
    # Specific modifier classes get +1 — beats generic parking but loses to prohibitions
    _MODIFIER_BOOST = {
        "handicap", "ev_charging", "motorcycle", "truck", "trailer",
        "diagonal_parking", "parallel_parking", "residents",
        "parking_disc", "exception_plate", "arrow_plate",
    }
    for cls in _PROHIBIT_BOOST:
        if cls in votes:
            votes[cls] += 3
    for cls in _MODIFIER_BOOST:
        if cls in votes:
            votes[cls] += 1

    top_score = max(votes.values())
    for kw, cls in _LABEL_MAP:
        if votes.get(cls, 0) == top_score:
            return cls

    return "parking"


def _is_blocklisted(label: str) -> bool:
    return any(kw in label.lower() for kw in _BLOCKLIST)


def _nms(boxes, scores, labels, iou_thr=0.45):
    if not boxes:
        return [], [], []
    arr = np.array(boxes, dtype=np.float32)
    sc  = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    areas = (x2-x1)*(y2-y1)
    order = sc.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(int(i))
        if order.size == 1: break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thr]
    return [boxes[k] for k in keep], [scores[k] for k in keep], [labels[k] for k in keep]


def _remove_contained(boxes, scores, labels, containment_thr=0.92):
    """
    Drop any box that is largely contained within a larger box.
    A box B is dropped if (intersection_area / B_area) > containment_thr,
    i.e. more than 82% of B sits inside a bigger box.
    Keeps the larger (enclosing) box.
    """
    if len(boxes) <= 1:
        return boxes, scores, labels

    arr   = np.array(boxes, dtype=np.float32)
    areas = (arr[:,2]-arr[:,0]) * (arr[:,3]-arr[:,1])
    keep  = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            # Check if box j is contained inside box i
            ix1 = max(arr[i,0], arr[j,0]); iy1 = max(arr[i,1], arr[j,1])
            ix2 = min(arr[i,2], arr[j,2]); iy2 = min(arr[i,3], arr[j,3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if areas[j] > 0 and inter / areas[j] > containment_thr:
                if areas[i] > areas[j]:   # i is bigger — drop j
                    print(f"[detector] containment drop {boxes[j]} inside {boxes[i]}")
                    keep[j] = False

    kb = [b for b, k in zip(boxes,  keep) if k]
    ks = [s for s, k in zip(scores, keep) if k]
    kl = [lb for lb, k in zip(labels, keep) if k]
    return kb, ks, kl


def _best_anchor(boxes, scores, roi_cx, roi_cy):
    """Highest score × area × centredness within ROI."""
    best_i, best_s = 0, -1.0
    for i, (box, sc) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        area = (x2-x1)*(y2-y1)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        dist  = ((cx-roi_cx)**2 + (cy-roi_cy)**2) ** 0.5
        norm  = ((roi_cx**2 + roi_cy**2) ** 0.5) or 1
        centr = 1.0 - min(dist/norm, 1.0)
        val   = sc * area * centr
        if val > best_s:
            best_s, best_i = val, i
    return best_i


def _pole_band_filter(boxes, scores, labels, anchor_box, roi_w, roi_h):
    """
    Keep boxes whose centre is within POLE_BAND_HALF_WIDTHS × anchor_half_width
    of the anchor centre X.  This is much tighter than the old edge-based check.
    """
    ax1, _, ax2, _ = anchor_box
    anchor_cx      = (ax1 + ax2) / 2
    anchor_hw      = max((ax2 - ax1) / 2, 1)
    max_dx         = anchor_hw * POLE_BAND_HALF_WIDTHS
    min_area       = roi_w * roi_h * MIN_BOX_AREA_FRACTION

    kb, ks, kl = [], [], []
    for b, s, lb in zip(boxes, scores, labels):
        x1, y1, x2, y2 = b
        cx   = (x1+x2) / 2
        area = (x2-x1)*(y2-y1)
        w    = x2 - x1
        h    = y2 - y1

        # Drop boxes that are too tall relative to width — these are merged
        # multi-sign clusters, not individual plates. Individual signs are
        # roughly square (aspect ratio < 3). A merged cluster of 5 signs
        # stacked vertically will have aspect ratio > 4.
        if h > 0 and (h / max(w, 1)) > 3.5:
            print(f"[detector] aspect-ratio drop ({x1},{y1},{x2},{y2}) h/w={h/max(w,1):.1f}")
            continue

        if abs(cx - anchor_cx) > max_dx:
            print(f"[detector] pole-band drop ({x1},{y1},{x2},{y2}) "
                  f"cx={cx:.0f} anchor_cx={anchor_cx:.0f} max_dx={max_dx:.0f}")
            continue
        if area < min_area:
            print(f"[detector] size-gate drop ({x1},{y1},{x2},{y2}) area={area:.0f}")
            continue

        kb.append(b); ks.append(s); kl.append(lb)
    return kb, ks, kl


def _apply_roi(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop to ROI. Returns (crop, origin_x, origin_y)."""
    H, W = img.shape[:2]
    x1 = int(W * ROI_X_START)
    x2 = int(W * ROI_X_END)
    y1 = int(H * ROI_Y_START)
    y2 = int(H * ROI_Y_END)
    print(f"[detector] ROI: x=[{x1},{x2}] y=[{y1},{y2}]  ({x2-x1}×{y2-y1} of {W}×{H})")
    return img[y1:y2, x1:x2], x1, y1


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_signs(
    img_array: np.ndarray,
) -> tuple[Optional[np.ndarray], list[tuple[int,int,int,int]], list[str]]:
    """Detect all signs on the closest centred pole.

    Args:
        img_array: RGB uint8 (H, W, 3). Full portrait phone image.

    Returns:
        combined_crop  – vertical span crop of sign cluster (full-image coords).
        boxes          – (x1,y1,x2,y2) in full-image coordinates.
        classes        – internal class name per box.
    """
    if img_array is None or img_array.size == 0:
        return None, [], []

    H, W = img_array.shape[:2]

    # 1. Pre-crop
    roi, roi_ox, roi_oy = _apply_roi(img_array)
    roi_h, roi_w = roi.shape[:2]

    # 2. DINO
    processor, model = _get_model()
    device = next(model.parameters()).device
    image  = Image.fromarray(roi)
    inputs = processor(images=image, text=[LABELS], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"],
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(roi_h, roi_w)],
    )[0]

    # 3. Parse (ROI-local coords)
    raw_boxes, raw_scores, raw_labels = [], [], []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if _is_blocklisted(label):
            print(f"[detector] blocklisted {label!r}")
            continue
        x1,y1,x2,y2 = map(int, box.tolist())
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(roi_w,x2), min(roi_h,y2)
        raw_boxes.append((x1,y1,x2,y2))
        raw_scores.append(float(score))
        raw_labels.append(label)
        print(f"[detector] raw ({x1},{y1},{x2},{y2}) score={score:.2f} {label!r}")

    # 4. NMS → containment filter
    boxes, scores, labels = _nms(raw_boxes, raw_scores, raw_labels)
    boxes, scores, labels = _remove_contained(boxes, scores, labels)
    if not boxes:
        print("[detector] nothing detected.")
        return None, [], []

    # 5. Anchor + pole band (ROI coords)
    anchor_idx = _best_anchor(boxes, scores, roi_w/2, roi_h/2)
    anchor_box = boxes[anchor_idx]
    print(f"[detector] anchor: {anchor_box} label={labels[anchor_idx]!r}")
    boxes, scores, labels = _pole_band_filter(boxes, scores, labels, anchor_box, roi_w, roi_h)

    # 6. Translate to full-image coords
    full_boxes = [(x1+roi_ox, y1+roi_oy, x2+roi_ox, y2+roi_oy) for x1,y1,x2,y2 in boxes]

    for (x1,y1,x2,y2), sc, lb in zip(full_boxes, scores, labels):
        print(f"[detector] final ({x1},{y1},{x2},{y2}) score={sc:.2f} {lb!r}")

    if not full_boxes:
        return None, [], []

    sx1 = max(0, min(b[0] for b in full_boxes))
    sy1 = max(0, min(b[1] for b in full_boxes))
    sx2 = min(W, max(b[2] for b in full_boxes))
    sy2 = min(H, max(b[3] for b in full_boxes))
    combined = img_array[sy1:sy2, sx1:sx2]

    # Return raw DINO labels — classifier.py handles conversion
    return (combined if combined.size else None), full_boxes, labels


# ── Visualisation ──────────────────────────────────────────────────────────────

def draw_boxes(
    img_array: np.ndarray,
    boxes: list[tuple[int,int,int,int]],
    classes: Optional[list[str]] = None,
) -> np.ndarray:
    """Draw labelled boxes + ROI outline. Input & output RGB."""
    vis = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)
    H, W = img_array.shape[:2]

    # ROI boundary (orange) — shows exactly what DINO sees
    rx1, ry1 = int(W*ROI_X_START), int(H*ROI_Y_START)
    rx2, ry2 = int(W*ROI_X_END),   int(H*ROI_Y_END)
    cv2.rectangle(vis, (rx1,ry1), (rx2,ry2), (0,140,255), 2)

    for i, (x1,y1,x2,y2) in enumerate(boxes):
        label = f"Sign {i+1}"
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 3)
        cv2.putText(vis, label, (x1, max(y1-10,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)