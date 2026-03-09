import numpy as np
import cv2
import pytesseract
from PIL import Image



def extract_text(img_array: np.ndarray) -> str:
    """
    Extract text from a cropped sign strip using Tesseract OCR.

    Splits the image into horizontal bands and OCR-s each independently
    so mixed blue/white panels each get correct inversion.
    """
    h, w = img_array.shape[:2]
    scale = max(1, 800 // max(h, w, 1))
    if scale > 1:
        img_array = cv2.resize(img_array, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        h, w = img_array.shape[:2]

    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    band_h = max(60, h // 8)
    texts = []

    for y in range(0, h, band_h):
        band = gray[y:min(y + band_h, h)]
        if band.shape[0] < 15:
            continue
        if band.mean() < 128:
            band = cv2.bitwise_not(band)
        _, thresh = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = Image.fromarray(thresh)
        try:
            t = pytesseract.image_to_string(image, lang="swe+eng", config="--psm 7")
        except pytesseract.TesseractError:
            try:
                t = pytesseract.image_to_string(image, lang="eng", config="--psm 7")
            except pytesseract.TesseractError:
                continue
        line = t.strip()
        if line:
            texts.append(line)

    return "\n".join(texts).upper().strip()
