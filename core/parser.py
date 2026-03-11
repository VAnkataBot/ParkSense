"""Parse Swedish parking signs — multi-anchor model.

Pole reading model
──────────────────
Signs are passed in bottom-to-top order.  Each pole may have MULTIPLE
anchors (e.g. Lastplats + blue P).  Modifier plates belong to the anchor
directly above them (i.e. the next anchor encountered going bottom-to-top).

Anchor classes
──────────────
  parking / diagonal_parking / parallel_parking  → parking ALLOWED
  no_parking                                      → parking PROHIBITED
  no_stopping                                     → stopping PROHIBITED
  loading_zone                                    → loading only

Modifier classes (sub-plates that qualify the anchor above them)
────────────────────────────────────────────────────────────────
  exception_plate   → time/day window ("7-19", "Fred 0-6")
  distance_plate    → zone extent ("0-15 m")
  parking_disc      → disc required
  arrow_plate       → direction indicator (informational)
  handicap          → disabled badge only
  ev_charging       → EV only
  motorcycle        → motorcycles only
  truck             → trucks only
  trailer           → trailers only
  residents         → residents permit

"Övrig tid" logic
──────────────────
If a blue P sign's OCR text contains "ÖVRIG TID" (or "OVRIG TID"), that
anchor's active window is the COMPLEMENT of all other anchors' windows.
We compute this by collecting all restricted intervals from the other
anchors and inverting them.

Swedish time convention
───────────────────────
  7-19        → Mon–Fri
  (11-17)     → Sat / day before holiday
  Sunday      → always free unless explicitly stated
  Fred 0-6    → Friday 00:00–06:00 (street cleaning)
"""

import re
from datetime import datetime
from typing import Optional

# ── Class sets ────────────────────────────────────────────────────────────────

ANCHOR_ALLOW    = {"parking", "parking_ovrig", "diagonal_parking", "parallel_parking"}
ANCHOR_PROHIBIT = {"no_parking", "no_stopping"}
ANCHOR_LOADING  = {"loading_zone"}
ANCHOR_CLASSES  = ANCHOR_ALLOW | ANCHOR_PROHIBIT | ANCHOR_LOADING

MODIFIER_NOTES = {
    "handicap":     "Disabled permit holders only.",
    "ev_charging":  "Electric vehicles (with charging) only.",
    "truck":        "Heavy vehicles (>3.5 t) only.",
    "motorcycle":   "Motorcycles / mopeds only.",
    "trailer":      "Trailers only.",
    "parking_disc": "Parking disc required — set to next half-hour on arrival.",
    "residents":    "Residents/permit holders only.",
}

# ── Day map ───────────────────────────────────────────────────────────────────

DAY_MAP = {
    "MÅN": 0, "MON": 0,
    "TIS": 1, "TUE": 1,
    "ONS": 2, "WED": 2,
    "TOR": 3, "TORS": 3, "THU": 3,
    "FRE": 4, "FRED": 4, "FRI": 4,
    "LÖR": 5, "SAT": 5,
    "SÖN": 6, "SÖ": 6, "SUN": 6,   # SÖ is the common short form on signs
}

# Swedish day names indexed 0–6 for readable messages
_DAY_NAMES = {0: "Mån", 1: "Tis", 2: "Ons", 3: "Tor", 4: "Fre", 5: "Lör", 6: "Sön"}

# ── Text helpers ──────────────────────────────────────────────────────────────

_TIME_RE = r"(\d{1,2}[.:,]?\d{0,2})\s*[-–]\s*(\d{1,2}[.:,]?\d{0,2})"

# Day names sorted longest-first so "TORS" beats "TOR", "FRED" beats "FRE" etc.
_DAY_PAT = "(?:" + "|".join(
    re.escape(k) for k in sorted(DAY_MAP.keys(), key=len, reverse=True)
) + ")"


def _strip_day_times(text: str) -> str:
    """Remove patterns like 'FRED 0-6' or 'TORS 8-18' so they don't
    interfere with the general weekday time search."""
    return re.sub(
        rf"\b{_DAY_PAT}\b\s*\d{{1,2}}[.:,]?\d{{0,2}}\s*[-–]\s*\d{{1,2}}[.:,]?\d{{0,2}}",
        "", text,
    )


def _norm(text: str) -> str:
    t = text.upper()
    t = re.sub(r"[|\\]", "", t)
    t = re.sub(r"[~_]", "-", t)
    t = re.sub(r"[^\w\s:()\-–./,]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_mins(s: str) -> int:
    s = s.replace(".", ":").replace(",", ":")
    p = s.split(":")
    return int(p[0]) * 60 + (int(p[1]) if len(p) > 1 else 0)


def _weekday_time(text: str) -> Optional[tuple[int, int]]:
    t = re.sub(r"\b0\s*[-–]\s*\d+\s*M+\b", "", text)
    t = re.sub(r"\([^)]*\)", "", t)
    t = _strip_day_times(t)          # don't pick up "FRED 0-6" as the weekday time
    m = re.search(_TIME_RE, t)
    return (_parse_mins(m.group(1)), _parse_mins(m.group(2))) if m else None


def _saturday_time(text: str) -> Optional[tuple[int, int]]:
    m = re.search(r"\(\s*" + _TIME_RE + r"\s*\)", text)
    return (_parse_mins(m.group(1)), _parse_mins(m.group(2))) if m else None


def _single_day(text: str) -> Optional[int]:
    for name, idx in DAY_MAP.items():
        if re.search(r"\b" + name + r"\b", text):
            return idx
    return None


def _zone_metres(text: str) -> Optional[int]:
    m = re.search(r"0\s*[-–]\s*(\d+)\s*M+\b", text)
    return int(m.group(1)) if m else None


def _fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _is_ovrig_tid(text: str) -> bool:
    return "ÖVRIG TID" in text or "OVRIG TID" in text


# ── Interval helpers (minutes since midnight, within one day) ─────────────────

def _intervals_for_anchor(
    mod_text: str, weekday: int, anchor_cls: str = ""
) -> list[tuple[int, int]]:
    """
    Return list of (start, end) minute intervals when the anchor's rule
    is ACTIVE today.  Empty list = active at all times (no restriction).

    For ANCHOR_ALLOW signs: day-specific times (e.g. "FRED 0-6") are
    PROHIBITION overlays, not allowed windows — they are handled separately
    by _day_prohibition_intervals and are intentionally excluded here.
    For ANCHOR_PROHIBIT/LOADING: day-specific time = when prohibition is active.
    """
    day = _single_day(mod_text)

    if day is not None:
        m = re.search(rf"\b{_DAY_PAT}\b\s*{_TIME_RE}", mod_text)
        if m and weekday == day and anchor_cls not in ANCHOR_ALLOW:
            # PROHIBIT/LOADING: return the day-specific active window
            return [(_parse_mins(m.group(1)), _parse_mins(m.group(2)))]
        # ALLOW: day+time is a prohibition overlay — fall through to general time
        # Any other day: also fall through to general time

    # General time (day-specific patterns stripped inside _weekday_time)
    t_wd  = _weekday_time(mod_text)
    t_sat = _saturday_time(mod_text)

    if not t_wd and not t_sat:
        return []   # no time info → always active

    if weekday == 6:
        return []   # Sunday → free unless explicitly restricted

    if weekday == 5:
        return [t_sat] if t_sat else []

    # Mon–Fri
    return [t_wd] if t_wd else []


def _day_prohibition_intervals(mod_text: str, weekday: int) -> list[tuple[int, int]]:
    """Prohibition windows from day-specific plates on a blue P sign.
    E.g. 'FRED 0-6' = no parking Friday 00:00–06:00 (street cleaning)."""
    day = _single_day(mod_text)
    if day is None or day != weekday:
        return []
    m = re.search(rf"\b{_DAY_PAT}\b\s*{_TIME_RE}", mod_text)
    if m:
        return [(_parse_mins(m.group(1)), _parse_mins(m.group(2)))]
    return []


def _in_any(intervals: list[tuple[int, int]], minutes: int) -> bool:
    return any(s <= minutes < e for s, e in intervals)


def _complement(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the time periods NOT covered by the given intervals (within 0–1440)."""
    if not intervals:
        return []   # original covers nothing → complement is everything (handled separately)
    merged = sorted(intervals)
    result = []
    cursor = 0
    for s, e in merged:
        if cursor < s:
            result.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < 1440:
        result.append((cursor, 1440))
    return result


# ── Grouping ──────────────────────────────────────────────────────────────────

def _group_signs(signs: list[dict]) -> list[dict]:
    """
    Group signs bottom-to-top into anchor+modifiers blocks.

    Returns list of:
        {
            "anchor_cls":  str,
            "anchor_text": str,
            "mod_text":    str,   # merged text of all modifier plates
            "notes":       [str]  # human-readable modifier notes
        }
    Ordered bottom-to-top (index 0 = bottom anchor).
    """
    groups: list[dict] = []
    current_mods: list[dict] = []

    for sign in signs:   # already bottom-to-top
        cls  = (sign.get("class") or "").lower()
        text = _norm(sign.get("text") or "")

        if cls in ANCHOR_CLASSES:
            # Start a new group — attach accumulated modifiers to THIS anchor
            groups.append({
                "anchor_cls":  cls,
                "anchor_text": text,
                "mod_text":    " ".join(_norm(m.get("text") or "") for m in current_mods),
                "mods":        list(current_mods),
            })
            current_mods = []
        else:
            # Modifier plate — accumulate until we hit an anchor
            current_mods.append(sign)

    # Any leftover modifiers above the topmost anchor belong to it
    if current_mods and groups:
        extra = " ".join(_norm(m.get("text") or "") for m in current_mods)
        groups[-1]["mod_text"] = (groups[-1]["mod_text"] + " " + extra).strip()
        groups[-1]["mods"].extend(current_mods)

    return groups


def _build_notes(group: dict) -> list[str]:
    notes: list[str] = []
    combined = group["anchor_text"] + " " + group["mod_text"]

    # Anchor class itself may carry a note (e.g. handicap as anchor)
    anchor = group["anchor_cls"]
    if anchor in MODIFIER_NOTES:
        notes.append(MODIFIER_NOTES[anchor])

    for mod in group.get("mods", []):
        cls = (mod.get("class") or "").lower()
        if cls in MODIFIER_NOTES and MODIFIER_NOTES[cls] not in notes:
            notes.append(MODIFIER_NOTES[cls])

    zone = _zone_metres(combined)
    if zone:
        notes.append(f"Reserved zone: 0–{zone} m.")

    if any(kw in combined for kw in ("AVGIFT", "TAXA", "BETALA DIGITALT", "PARKERING.STOCKHOLM")):
        if "PARKERING.STOCKHOLM" in combined or "BETALA DIGITALT" in combined:
            notes.append("Pay digitally — parkering.stockholm.se or Stockholm parking app (no meter).")
        else:
            notes.append("Paid parking — check meter/app for tariff.")

    if any(kw in combined for kw in ("BOENDE", "TILLSTÅND", "TILLSTAND")):
        day = _single_day(combined)
        if day == 6:
            notes.append("Sunday: residents/permit holders only.")
        else:
            notes.append("Residents/permit holders only.")

    if "P-SKIVA" in combined or "PSKIVA" in combined:
        notes.append("Parking disc required.")

    return notes


# ── Public API ────────────────────────────────────────────────────────────────

def parse_parking_rules(
    signs: list[dict],
    now: datetime,
) -> dict:
    """Parse an ordered sign list (bottom-to-top) into a verdict.

    Args:
        signs:  List of {"class": str, "text": str}, bottom-to-top.
        now:    Current datetime.

    Returns:
        {
            "can_park": True | False | None,
            "message":  str,
            "notes":    [str],
            "groups":   [debug info per anchor group]
        }
    """
    if not signs:
        return {"can_park": None, "message": "No signs detected.", "notes": [], "groups": []}

    weekday = now.weekday()
    minutes = now.hour * 60 + now.minute
    groups  = _group_signs(signs)

    if not groups:
        # No anchor detected — but signs were found. In Swedish law, vehicle-restriction
        # symbols (handicap, EV, etc.) always accompany a blue P. Infer one.
        _VEHICLE_MODIFIERS = {"handicap", "ev_charging", "motorcycle", "truck",
                              "trailer", "residents", "parking_disc"}
        if any((s.get("class") or "") in _VEHICLE_MODIFIERS for s in signs):
            groups = [{
                "anchor_cls":  "parking",
                "anchor_text": "",
                "mod_text":    " ".join(_norm(s.get("text") or "") for s in signs),
                "mods":        list(signs),
            }]
        else:
            return {"can_park": None, "message": "No anchor sign found.", "notes": [], "groups": []}

    # ── Evaluate each group ───────────────────────────────────────────────────
    # Collect all non-övrig-tid restricted intervals first (needed for complement)
    all_restricted_intervals: list[tuple[int, int]] = []
    evaluated: list[dict] = []

    for g in groups:
        cls      = g["anchor_cls"]
        mod_text = g["mod_text"]
        notes    = _build_notes(g)
        ovrig    = cls == "parking_ovrig" or _is_ovrig_tid(g["anchor_text"] + " " + mod_text)
        intervals = _intervals_for_anchor(mod_text, weekday, cls)

        evaluated.append({
            "cls":       cls,
            "mod_text":  mod_text,
            "notes":     notes,
            "ovrig":     ovrig,
            "intervals": intervals,
            "anchor_text": g["anchor_text"],
        })

        if not ovrig and cls in (ANCHOR_PROHIBIT | ANCHOR_LOADING):
            if intervals:
                all_restricted_intervals.extend(intervals)
            else:
                # No time restriction on a prohibition = all day restricted
                all_restricted_intervals.append((0, 1440))

    # ── Determine verdict ─────────────────────────────────────────────────────
    verdict:     Optional[bool] = None
    verdict_msg: str = ""
    all_notes:   list[str] = []

    for g in evaluated:
        cls       = g["cls"]
        intervals = g["intervals"]
        ovrig     = g["ovrig"]
        notes     = g["notes"]
        all_notes.extend(notes)

        if ovrig and cls in ANCHOR_ALLOW:
            # P sign with "övrig tid" — active when OTHER anchors are NOT active
            free_intervals = _complement(all_restricted_intervals)
            if not all_restricted_intervals:
                # No other restrictions → always allowed
                verdict, verdict_msg = True, "Parking allowed (no restrictions)."
            elif _in_any(free_intervals, minutes):
                verdict, verdict_msg = True, "Parking allowed — övrig tid (outside loading/restricted hours)."
            else:
                verdict, verdict_msg = False, "No parking now — loading zone or restriction active."
            continue

        if cls in ANCHOR_LOADING:
            if not intervals or _in_any(intervals, minutes):
                verdict, verdict_msg = False, "Loading zone — no parking now."
            else:
                # Outside loading hours — don't override if P sign already set True
                if verdict is not True:
                    verdict, verdict_msg = True, "Outside loading zone hours — parking allowed."
            continue

        if cls in ANCHOR_PROHIBIT:
            if not intervals:
                verdict, verdict_msg = False, "No stopping/parking at all times."
            elif _in_any(intervals, minutes):
                day = _single_day(g["mod_text"])
                if day is not None:
                    label = f"{_DAY_NAMES.get(day, '')} {_fmt(intervals[0][0])}–{_fmt(intervals[0][1])}"
                    verdict, verdict_msg = False, f"No parking — street cleaning {label}."
                else:
                    label = f"{_fmt(intervals[0][0])}–{_fmt(intervals[0][1])}"
                    verdict, verdict_msg = False, f"Parking prohibited {label}."
            else:
                if verdict is not True:
                    verdict, verdict_msg = True, "Outside restricted hours — parking allowed."
            continue

        if cls in ANCHOR_ALLOW and not ovrig:
            # Day-specific prohibition overlay (e.g. "FRED 0-6" on blue P = street cleaning)
            prohib = _day_prohibition_intervals(g["mod_text"], weekday)
            if prohib and _in_any(prohib, minutes):
                p = prohib[0]
                label = f"{_DAY_NAMES[weekday]} {_fmt(p[0])}–{_fmt(p[1])}"
                verdict, verdict_msg = False, f"No parking — street cleaning {label}."
                continue

            t_wd  = _weekday_time(g["mod_text"])
            t_sat = _saturday_time(g["mod_text"])
            if not intervals and not t_wd and not t_sat:
                if verdict is None:   # don't overwrite a more specific verdict
                    verdict, verdict_msg = True, "Parking allowed (no time restriction)."
            elif _in_any(intervals, minutes):
                label = f"{_fmt(intervals[0][0])}–{_fmt(intervals[0][1])}" if intervals else ""
                if verdict is not False:
                    verdict, verdict_msg = True, f"Parking allowed{' — ' + label if label else ''}."
            else:
                if verdict is not False:
                    verdict, verdict_msg = False, "Outside allowed parking hours."

    if verdict is None:
        verdict_msg = "Could not determine parking rules — check signs manually."

    return {
        "can_park": verdict,
        "message":  verdict_msg,
        "notes":    list(dict.fromkeys(all_notes)),
        "groups": [
            {
                "anchor":   g["cls"],
                "ovrig_tid": g["ovrig"],
                "mod_text": g["mod_text"],
                "intervals": g["intervals"],
            }
            for g in evaluated
        ],
    }