"""Classify a detected sign box into a semantic class.

Priority order
──────────────
1. OCR text rules  — unambiguous Swedish keywords (most reliable)
2. DINO label      — for symbol-only plates with no text
3. Fallback        — "unknown"

This module is intentionally separate from both the detector (which finds
boxes) and the parser (which interprets rules).  It is the only place that
knows about Swedish sign vocabulary.
"""

import re

# ── Text normalisation ────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[|\\]", "", t)
    t = re.sub(r"[~_]", "-", t)
    t = re.sub(r"[^\w\s:()\-–./,]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ── OCR text → class rules ────────────────────────────────────────────────────
# Each entry: (pattern_or_list_of_patterns, class_name)
# List = ALL patterns must match.  String = single regex or keyword.
# Checked in order — first match wins.  More specific rules first.

_OCR_RULES: list[tuple] = [

    # ── Hard prohibitions ─────────────────────────────────────────────────
    ("STOPFÖRBUD",                          "no_stopping"),
    ("STOP FÖRBUD",                         "no_stopping"),
    ("PARKERING FÖRBJUDEN",                 "no_parking"),
    ("PARKERINGSFÖRBUD",                    "no_parking"),

    # ── Loading zone ──────────────────────────────────────────────────────
    ("LASTPLATS",                           "loading_zone"),
    ("LAST PLATS",                          "loading_zone"),

    # ── Övrig tid — must come before generic parking ──────────────────────
    ("ÖVRIG TID",                           "parking_ovrig"),
    ("OVRIG TID",                           "parking_ovrig"),

    # ── Parking orientation variants ──────────────────────────────────────
    ("SNEDPARKERING",                       "diagonal_parking"),
    ("PARALLELLPARKERING",                  "parallel_parking"),
    ("PARALELLPARKERING",                   "parallel_parking"),

    # ── Residents / permit ────────────────────────────────────────────────
    ("BOENDE",                              "residents"),
    ("TILLSTÅND",                           "residents"),
    ("TILLSTAND",                           "residents"),

    # ── Payment info plates ───────────────────────────────────────────────
    ("BETALA DIGITALT",                     "payment_info"),
    ("PARKERING.STOCKHOLM",                 "payment_info"),

    # ── Disc required ─────────────────────────────────────────────────────
    ("P-SKIVA",                             "parking_disc"),
    ("PSKIVA",                              "parking_disc"),
    ("P SKIVA",                             "parking_disc"),

    # ── Distance plate — must come before time plate ──────────────────────
    (r"0\s*[-–]\s*\d+\s*M\b",              "distance_plate"),

    # ── Exception plates: day name + time range ───────────────────────────
    (["TORS",  r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["FRED",  r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["MÅN",   r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["TIS",   r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["ONS",   r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["LÖR",   r"\d+\s*[-–]\s*\d+"],       "exception_plate"),
    (["SÖN",   r"\d+\s*[-–]\s*\d+"],       "exception_plate"),

    # ── Exception plates: Avgift/Taxa + time range ────────────────────────
    (["AVGIFT", r"\d+\s*[-–]\s*\d+"],      "exception_plate"),
    (["TAXA",   r"\d+\s*[-–]\s*\d+"],      "exception_plate"),
]

# ── DINO label → class (symbol-only fallback) ─────────────────────────────────
# Only used when OCR text gives no clear class.
# Ordered most-specific first to avoid generic matches swallowing specific ones.

_DINO_SYMBOL_MAP: list[tuple[str, str]] = [
    ("no stopping",      "no_stopping"),
    ("no parking",       "no_parking"),
    ("loading zone",     "loading_zone"),
    ("lastplats",        "loading_zone"),
    ("handicap",         "handicap"),
    ("wheelchair",       "handicap"),
    ("electric vehicle", "ev_charging"),
    ("charging",         "ev_charging"),
    ("motorcycle",       "motorcycle"),
    ("diagonal",         "diagonal_parking"),   # before truck — diagonal beats truck on tie
    ("parallel",         "parallel_parking"),
    ("truck",            "truck"),
    ("trailer parking sign", "trailer"),
    ("parking disc",     "parking_disc"),
    ("arrow",            "arrow_plate"),
    ("residents",        "residents"),
    ("blue parking sign","parking"),
    ("parking sign",     "parking"),
    ("parking",          "parking"),
]


def _matches(rule, text: str) -> bool:
    if isinstance(rule, list):
        return all(bool(re.search(p, text)) for p in rule)
    return bool(re.search(rule, text))


def classify(ocr_text: str, dino_label: str, symbol_cls: str | None = None) -> str:
    """Classify a single detected sign box.

    Args:
        ocr_text:   Raw OCR text from the sign crop (any case).
        dino_label: Raw label string from Grounding DINO.
        symbol_cls: Optional class from CLIP symbol classifier (empty-OCR crops only).

    Returns:
        Semantic class string e.g. "loading_zone", "parking",
        "no_stopping", "exception_plate", "handicap", etc.
        Returns "unknown" if nothing matches.
    """
    text = _norm(ocr_text)
    dino = (dino_label or "").lower()

    # ── 0. CLIP symbol classifier — highest trust for symbol-only plates ──
    if symbol_cls and not text:
        return symbol_cls

    # ── 1. OCR text rules ─────────────────────────────────────────────────
    for rule, cls in _OCR_RULES:
        if _matches(rule, text):
            return cls

    # ── 2. Bare time plate: just hours with no other keywords ─────────────
    # e.g. "7-19" or "7-19 (11-17)" — short text, only numbers and dashes
    if re.match(r"^\s*\d{1,2}\s*[-–]\s*\d{1,2}", text) and len(text) < 25:
        return "exception_plate"

    # ── 3. DINO label fallback — symbol-only plates ───────────────────────
    # If DINO says "handicap parking sign" / "truck parking sign" etc. AND
    # OCR found actual text (e.g. "P"), it's a combined blue-P + symbol sign
    # → treat as parking anchor so the parser finds an anchor.
    # If OCR is empty, it's the symbol plate alone (wheelchair icon, EV icon etc.)
    # → fall through to the DINO map so it becomes a modifier class (handicap etc.).
    _VEHICLE_KEYWORDS = {"handicap", "wheelchair", "electric vehicle", "charging",
                         "motorcycle", "truck", "trailer", "residents"}
    if ("parking sign" in dino or "parking" in dino) and text:
        for kw in _VEHICLE_KEYWORDS:
            if kw in dino:
                return "parking"

    for keyword, cls in _DINO_SYMBOL_MAP:
        if keyword in dino:
            return cls

    # ── 4. Any digit-dash-digit in OCR = time plate of some kind ─────────
    if re.search(r"\d+\s*[-–]\s*\d+", text):
        return "exception_plate"

    return "unknown"


def classify_signs(signs: list[dict]) -> list[dict]:
    """Add/overwrite 'class' on each sign dict using OCR + DINO label.

    Each sign dict should have:
        "text":       str  — OCR text from that box
        "dino_label": str  — raw DINO label for that box

    Returns the same list with "class" added to each item.
    """
    for sign in signs:
        sign["class"] = classify(
            sign.get("text") or "",
            sign.get("dino_label") or "",
            sign.get("symbol_cls") or None,
        )
    return signs