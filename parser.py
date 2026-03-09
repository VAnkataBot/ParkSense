import re
from typing import Optional, Tuple, List
from datetime import datetime

# Swedish day name mappings
DAY_MAP = {
    "MÅN": 0, "MON": 0,
    "TIS": 1, "TUE": 1,
    "ONS": 2, "WED": 2,
    "TOR": 3, "THU": 3,
    "FRE": 4, "FRI": 4,
    "LÖR": 5, "SAT": 5,
    "SÖN": 6, "SUN": 6,
}

# YOLO class labels that mean "no parking regardless of time"
NO_PARKING_CLASSES = {
    "round red circle swedish no parking prohibited sign",
    "round red circle swedish area-wide parking prohibition zone sign",
}

# YOLO class labels that restrict who can park
RESTRICTED_TO_CLASSES = {
    "square blue plate with international wheelchair accessibility symbol white icon": "Disabled badge holders only.",
    "rectangular white plate with heavy truck lorry vehicle silhouette icon": "Heavy goods vehicles (>3.5t) only.",
    "rectangular white plate with passenger car sedan vehicle silhouette icon": "Passenger cars only.",
    "rectangular white plate with motorcycle moped vehicle silhouette icon": "Motorcycles/mopeds only.",
    "rectangular white plate with trailer caravan vehicle silhouette icon": "Trailers only.",
    "rectangular white plate with electric vehicle charging plug lightning bolt icon": "Electric vehicles with charging capability only.",
    "rectangular white plate with permit text swedish special permit required parking sign": "Special permit required to park.",
    "rectangular white plate with tenant or boende text swedish residents only parking sign": "Paying tenants only.",
    "rectangular white plate with visitor or customer text swedish visitors only parking sign": "Visitors/customers only.",
}

# YOLO class labels that mean pay to park
PAY_CLASSES = {"rectangular white plate with coin or fee text swedish paid parking sign"}


def _normalize(text: str) -> str:
    """Clean up common OCR errors before parsing."""
    text = text.upper()
    text = re.sub(r"[|\\]", "", text)
    text = re.sub(r"~", "-", text)
    text = re.sub(r"[^\w\s:()\-–./,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_time(s: str) -> int:
    """Parse time string like '8', '08', '8.00', '8:00' into minutes since midnight."""
    s = s.replace(".", ":").replace(",", ":")
    parts = s.split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    return hours * 60 + minutes


def _parse_day_range(text: str) -> Optional[Tuple[int, int]]:
    """Extract day range from text, e.g. 'MÅN-FRE' → (0, 4)."""
    pattern = r"(" + "|".join(DAY_MAP.keys()) + r")\s*[-–]\s*(" + "|".join(DAY_MAP.keys()) + r")"
    match = re.search(pattern, text)
    if match:
        return DAY_MAP[match.group(1)], DAY_MAP[match.group(2)]
    return None


def _parse_time_range(text: str) -> Optional[Tuple[int, int]]:
    """Extract time range from text, e.g. '8-17', '08.00-18.00' → (480, 1020)."""
    pattern = r"(\d{1,2}[.:,]?\d{0,2})\s*[-–]\s*(\d{1,2}[.:,]?\d{0,2})"
    match = re.search(pattern, text)
    if match:
        start = _parse_time(match.group(1))
        end = _parse_time(match.group(2))
        return start, end
    return None


def _parse_max_duration(text: str) -> Optional[int]:
    """Extract max parking duration in minutes, e.g. '0-13 M' → 13, '2 TIM' → 120."""
    # Match patterns like "0-13 M", "0– 13M", "2 TIM", "3 TIMMAR"
    match = re.search(r"0\s*[-–]\s*(\d+)\s*M", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*TIM", text)
    if match:
        return int(match.group(1)) * 60
    return None


def parse_parking_rules(text: str, now: datetime, detected_classes: Optional[List[str]] = None) -> dict:
    """
    Parse Swedish parking sign text + YOLO class labels and determine if parking is allowed.

    Args:
        text: OCR text from the sign (will be normalised internally).
        now: Current datetime to evaluate against.
        detected_classes: List of YOLO class name strings detected on the sign.

    Returns:
        dict with keys:
            can_park (bool): Whether parking is currently allowed.
            message (str): Human-readable explanation.
            notes (list): Additional conditions (pay, disabled-only, duration, etc.)
    """
    detected_classes = [c.lower() for c in (detected_classes or [])]
    text = _normalize(text)
    notes = []

    # 1. Hard no-parking from YOLO class
    for cls in detected_classes:
        if cls in NO_PARKING_CLASSES:
            return {"can_park": False, "message": "Parking prohibited.", "notes": []}

    # 2. Hard no-parking from OCR text
    no_parking_keywords = ["FÖRBUD", "PARKERING FÖRBJUDEN", "STOPFÖRBUD"]
    for kw in no_parking_keywords:
        if kw in text:
            return {"can_park": False, "message": "Parking prohibited (no parking sign).", "notes": []}

    # 3. Collect restriction notes from symbol classes
    for cls in detected_classes:
        if cls in RESTRICTED_TO_CLASSES:
            notes.append(RESTRICTED_TO_CLASSES[cls])
        if cls in PAY_CLASSES:
            notes.append("Paid parking — check meter/app for tariff.")

    # 4. Parse time and day restrictions from OCR
    current_weekday = now.weekday()
    current_minutes = now.hour * 60 + now.minute

    day_range = _parse_day_range(text)
    time_range = _parse_time_range(text)
    max_duration = _parse_max_duration(text)

    if max_duration:
        if max_duration < 60:
            notes.append(f"Maximum parking duration: {max_duration} minutes.")
        elif max_duration % 60 == 0:
            hours = max_duration // 60
            notes.append(f"Maximum parking duration: {hours} hour{'s' if hours != 1 else ''}.")
        else:
            notes.append(f"Maximum parking duration: {max_duration // 60}h {max_duration % 60}min.")

    # Check AVGIFT (paid parking) in text
    if "AVGIFT" in text:
        notes.append("Paid parking — check meter/app for tariff.")

    day_restricted = True
    if day_range:
        start_day, end_day = day_range
        if start_day <= end_day:
            day_restricted = start_day <= current_weekday <= end_day
        else:
            day_restricted = current_weekday >= start_day or current_weekday <= end_day

    time_restricted = True
    if time_range:
        start_min, end_min = time_range
        time_restricted = start_min <= current_minutes <= end_min

    restricted_now = day_restricted and time_restricted

    day_str = ""
    if day_range:
        days = list(DAY_MAP.keys())
        day_str = f" on {days[day_range[0]*2]}-{days[day_range[1]*2]}"

    time_str = ""
    if time_range:
        s, e = time_range
        time_str = f" between {s//60:02d}:{s%60:02d} and {e//60:02d}:{e%60:02d}"

    if restricted_now:
        return {
            "can_park": False,
            "message": f"Restrictions apply{day_str}{time_str}.",
            "notes": notes,
        }
    else:
        if day_range or time_range:
            message = f"Outside restricted hours{day_str}{time_str}. Parking allowed."
        else:
            message = "No time restrictions detected. Parking allowed."
        return {"can_park": True, "message": message, "notes": notes}
