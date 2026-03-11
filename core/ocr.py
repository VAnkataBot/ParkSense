"""OCR text extraction using EasyOCR (Swedish + English).

Two-pass strategy:
  Pass 1 — standard image (black/dark text on white/light background)
  Pass 2 — red-channel mask (catches Sunday/holiday times printed in red)
Results are merged, deduped, and returned as a single uppercase string.
"""

import cv2
import numpy as np
import easyocr

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["sv", "en"], gpu=True)
    return _reader


def _red_mask(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return a high-contrast grayscale image with only red pixels kept,
    or None if the crop contains no significant red content."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red wraps around 0/180 in HSV
    lo1, hi1 = np.array([0,  100, 80]),  np.array([10,  255, 255])
    lo2, hi2 = np.array([165, 100, 80]), np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))

    if cv2.countNonZero(mask) < 30:   # not enough red pixels
        return None

    # Render red pixels as black on white — easy for OCR
    result = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)
    result[mask > 0] = 0
    # Dilate slightly to thicken thin strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.dilate(result, kernel, iterations=1)


def extract_text(img_array: np.ndarray) -> str:
    """Extract all text from a cropped sign image, returned uppercase.

    Runs a second OCR pass on red-masked pixels to catch Sunday/holiday
    times that are printed in red and invisible to standard OCR.
    """
    reader  = _get_reader()
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Pass 1 — normal image
    pass1: list[str] = reader.readtext(img_array, detail=0, paragraph=False)  # type: ignore[assignment]

    # Pass 2 — red text only
    pass2: list[str] = []
    red_img = _red_mask(img_bgr)
    if red_img is not None:
        pass2 = reader.readtext(red_img, detail=0, paragraph=False)  # type: ignore[assignment]

    # Merge: pass1 lines first, then any pass2 lines not already present
    pass1_upper = [t.upper().strip() for t in pass1 if t.strip()]
    pass2_upper = [t.upper().strip() for t in pass2 if t.strip()]
    combined    = pass1_upper + [t for t in pass2_upper if t not in pass1_upper]

    return "\n".join(combined).strip()
