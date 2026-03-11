"""CLIP-based classifier for symbol-only parking sign crops.

Used as a second pass when OCR is empty and DINO label is unreliable.
Only runs on crops where no text was detected — keeps latency down.

The CLIP ViT-B/32 model is ~340 MB and is downloaded once on first use
to ~/.cache/clip/.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# (prompt, class_name) — prompts tuned for Swedish blue sign symbols
_SYMBOL_LABELS: list[tuple[str, str]] = [
    ("motorcycle parking only sign white motorbike silhouette blue",   "motorcycle"),
    ("electric vehicle EV charging parking only sign blue plug car",   "ev_charging"),
    ("disabled handicap parking only sign white wheelchair blue",      "handicap"),
    ("truck heavy vehicle parking only sign blue white lorry",         "truck"),
    ("trailer caravan parking only sign blue white trailer",           "trailer"),
    ("parking time limit disc clock sign white clock blue",            "parking_disc"),
    ("blue parking sign white letter P",                               "parking"),
    ("no parking prohibition sign red circle diagonal stripe",         "no_parking"),
    ("no stopping prohibition sign red circle two diagonal stripes",   "no_stopping"),
    ("loading zone lastplats sign yellow background red cross",        "loading_zone"),
]

_TEXTS   = [t for t, _ in _SYMBOL_LABELS]
_CLASSES = [c for _, c in _SYMBOL_LABELS]

_model         = None
_preprocess    = None
_text_features = None


def _get_model():
    global _model, _preprocess, _text_features
    if _model is None:
        print("[symbol_classifier] Loading CLIP ViT-B/32…")
        import clip  # openai/clip — already in venv
        _model, _preprocess = clip.load("ViT-B/32", device="cpu")
        _model.eval()
        tokens = clip.tokenize(_TEXTS)
        with torch.no_grad():
            feats = _model.encode_text(tokens)
            _text_features = F.normalize(feats, dim=-1)
        print("[symbol_classifier] CLIP ready.")
    return _model, _preprocess, _text_features


def classify_symbol(image_crop: np.ndarray, cos_threshold: float = 0.20) -> str | None:
    """Run CLIP on a symbol-only crop.

    Uses raw cosine similarity (not softmax) so that close label distributions
    don't wash out the best match.  Returns top-1 class if its cosine similarity
    to the image is >= cos_threshold, else None.
    """
    if image_crop is None or image_crop.size == 0:
        return None

    model, preprocess, text_features = _get_model()

    pil_img    = Image.fromarray(image_crop)
    img_tensor = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat = F.normalize(img_feat, dim=-1)
        sims     = (img_feat @ text_features.T).squeeze(0)   # raw cosine similarities
        idx      = int(sims.argmax())
        best_sim = float(sims[idx])

    print(f"[symbol_classifier] best={_CLASSES[idx]!r} cos_sim={best_sim:.3f}")
    return _CLASSES[idx] if best_sim >= cos_threshold else None
