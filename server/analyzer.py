"""
Parking sign analysis — cross-platform inference stack.

Tier 1 — mlx-vlm      macOS Apple Silicon only (fastest, zero cost)
Tier 2 — Ollama        any OS with a local GPU/CPU (free, cross-platform)
Tier 3 — HF cloud      universal fallback (requires HF_TOKEN)

Environment variables:
  LLM_MODEL      mlx-community model ID   (default: mlx-community/Qwen3-VL-8B-Instruct-4bit)
  OLLAMA_MODEL   Ollama model tag         (default: qwen2.5vl:7b)
  OLLAMA_URL     Ollama base URL          (default: http://localhost:11434)
  HF_TOKEN       HF access token          (enables cloud fallback)
  HF_MODEL       HF model ID              (default: Qwen/Qwen2.5-VL-72B-Instruct)
"""

import os
import sys
import json
import base64
import logging
import tempfile
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

LOCAL_MODEL_ID = os.getenv("LLM_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

HF_TOKEN    = os.getenv("HF_TOKEN")
HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_ID = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

# Detect platform once at import time
_IS_APPLE_SILICON = sys.platform == "darwin" and os.uname().machine == "arm64"

VEHICLE_LABELS = {
    "car":        "a regular passenger car",
    "motorcycle": "a motorcycle/moped",
    "ev":         "an electric vehicle (EV)",
    "truck":      "a truck/light goods vehicle",
    "bus":        "a bus/coach",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(day: str, time: str, date: str, vehicle_type: str,
                 is_disabled: bool, has_resident_permit: bool, resident_zone: str) -> str:
    vehicle_desc = VEHICLE_LABELS.get(vehicle_type, "a regular passenger car")
    driver_desc = f"a driver in {vehicle_desc}"
    if is_disabled:
        driver_desc += " WITH a disability parking permit (♿)"
    if has_resident_permit:
        zone = f" zone {resident_zone}" if resident_zone else ""
        driver_desc += f", WITH a resident parking permit{zone}"

    return f"""You are an expert on Swedish parking signs and traffic regulations (Trafikförordningen).
Analyze this parking sign image for: {driver_desc}

Current time: {day} {time}
Current date: {date}

PHASE 1 — LIST every sign, plate, symbol, and text visible on the pole. Include:
- Main signs (round blue/red, P sign blue square)
- Supplementary plates (time ranges, arrows, text, symbols)
- Vehicle symbols (♿, EV, motorcycle, truck, taxi, etc.)
- Text plates (Boende, Tillstånd, Nyttotrafik, Avgift, Taxa, Övrig tid, etc.)
- Distance plates (e.g. "0-13 m"), arrow plates (↑ ↓ ↕)
If you find fewer than 3 items, look again.

PHASE 2 — RULES (in order):

STEP 1 — HARD BLOCKS. Check if ANY sign reserves the spot for a group this driver does NOT belong to:
- ♿ → disabled only. Block UNLESS driver has disability permit.
- EV charging → EVs only. Block UNLESS driver has EV.
- Motorcycle symbol → motorcycles only. Block UNLESS driver is on motorcycle.
- Boende/Boendeparkering → residents only. Block UNLESS driver has resident permit for the zone.
  Exception: "Boende Sö" only applies on Sundays — check today first.
- Tillstånd → permit holders only. Always block.
- Nyttotrafik → commercial vehicles only. Block UNLESS driver is in truck.
- Taxi → always block.
If blocked → can_park = false. STOP.

STEP 2 — TIMED RESTRICTIONS. For every no-parking/no-stopping sign with a time plate:
  Step A: Does the day match today ({day})? If NO → restriction inactive, skip.
  Step B: Is {time} within the stated hour range numerically? If NO → restriction inactive, skip.
  Only if BOTH yes → restriction active → can_park = false.

Example: "Torsd 0-6", today Thursday, time 09:30.
  Step A: Thursday = today → YES. Step B: is 09:30 in 00:00-06:00? 9 > 6 → NO.
  Restriction NOT active. Continue.

Time plate conventions:
  7-19           = weekdays Mon-Fri only
  (7-19)         = Saturdays only
  ((7-19))       = Sundays/holidays only
  No time plate  = 24/7

STEP 3 — P SIGN. Blue square P = parking allowed. Note: Avgift = fee, P-skiva = parking disc.

PHASE 3 — OUTPUT
Write signs and notes FIRST to complete reasoning before verdict.

Final check: is there a restriction active for THIS driver at {day} {time}?
- YES → can_park = false
- NO  → can_park = true (add fees/conditions to notes)
- Uncertain → can_park = null
Notes and can_park MUST be consistent.

Reply with ONLY valid JSON, no other text:
{{"signs": ["one short phrase each"], "notes": ["max 10 words each"], "can_park": true/false/null, "message": "max 15 words"}}"""


# ── Tier 1: mlx-vlm (macOS Apple Silicon only) ───────────────────────────────

_model     = None
_processor = None


def _load_local_model():
    global _model, _processor
    logger.info(f"Loading mlx model: {LOCAL_MODEL_ID}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    from huggingface_hub import snapshot_download
    from mlx_vlm import load

    local_path = os.path.join(MODEL_DIR, LOCAL_MODEL_ID.replace("/", "--"))
    if not os.path.isdir(local_path):
        logger.info(f"Downloading model to {local_path} ...")
        snapshot_download(LOCAL_MODEL_ID, local_dir=local_path)

    _model, _processor = load(local_path, trust_remote_code=True)
    logger.info("mlx model ready")


def _infer_mlx(image: Image.Image, prompt_text: str) -> str:
    global _model, _processor
    if _model is None:
        _load_local_model()

    from mlx_vlm import generate

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp, format="JPEG", quality=90)
        tmp_path = tmp.name

    try:
        if hasattr(_processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
            formatted_prompt = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt_text

        result = generate(
            _model, _processor,
            prompt=formatted_prompt,
            image=tmp_path,
            max_tokens=1024,
            temperature=0.1,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)
    finally:
        os.unlink(tmp_path)


# ── Tier 2: Ollama (cross-platform local) ────────────────────────────────────

def _ollama_available() -> bool:
    """Check if Ollama is running and has the requested model pulled."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
        return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def _infer_ollama(image_b64: str, prompt_text: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
        "messages": [{
            "role": "user",
            "content": prompt_text,
            "images": [image_b64],
        }],
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return data["message"]["content"].strip()


# ── Tier 3: HF cloud ──────────────────────────────────────────────────────────

def _infer_hf_cloud(image_b64: str, prompt_text: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model": HF_MODEL_ID,
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        HF_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"].strip()


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_image(
    image_b64: str,
    day: str,
    time: str,
    date: str,
    vehicle_type: str = "car",
    is_disabled: bool = False,
    has_resident_permit: bool = False,
    resident_zone: str = "",
) -> dict:
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    prompt_text = build_prompt(day, time, date, vehicle_type,
                               is_disabled, has_resident_permit, resident_zone)

    raw: str | None = None

    # Tier 1 — mlx-vlm (Apple Silicon only)
    if _IS_APPLE_SILICON:
        try:
            raw = _infer_mlx(image, prompt_text)
            logger.info("Inference via mlx-vlm")
        except Exception as e:
            logger.warning(f"mlx-vlm failed: {e}")

    # Tier 2 — Ollama (any OS, if running locally)
    if raw is None:
        if _ollama_available():
            try:
                raw = _infer_ollama(image_b64, prompt_text)
                logger.info(f"Inference via Ollama ({OLLAMA_MODEL})")
            except Exception as e:
                logger.warning(f"Ollama failed: {e}")
        else:
            logger.info("Ollama not available — skipping")

    # Tier 3 — HF cloud
    if raw is None:
        if not HF_TOKEN:
            raise RuntimeError(
                "No inference backend available. Options:\n"
                "  • macOS Apple Silicon: mlx-vlm runs automatically\n"
                f"  • Any OS: install Ollama and run: ollama pull {OLLAMA_MODEL}\n"
                "  • Any OS: set HF_TOKEN for cloud fallback"
            )
        logger.info(f"Inference via HF cloud ({HF_MODEL_ID})")
        raw = _infer_hf_cloud(image_b64, prompt_text)

    start = raw.index("{")
    end   = raw.rindex("}") + 1
    return json.loads(raw[start:end])
