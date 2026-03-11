import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime
from core.detector import detect_signs, draw_boxes
from core.color_detector import detect_signs_by_color
from core.ocr import extract_text
from core.classifier import classify_signs
from core.parser import parse_parking_rules
from core.symbol_classifier import classify_symbol

st.set_page_config(page_title="ParkSence", page_icon="🅿️", layout="centered")

with st.sidebar:
    st.header("Settings")
    dev_mode = st.toggle("Developer mode", value=False)

st.title("🅿️arkSence")
st.caption("Scan a Swedish parking sign to know if you can park.")

input_method = st.radio("Input method", ["Upload image", "Use camera"], horizontal=True)

image = None
if input_method == "Upload image":
    uploaded = st.file_uploader("Upload a photo of a parking sign", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
else:
    captured = st.camera_input("Point camera at a parking sign")
    if captured:
        image = Image.open(captured).convert("RGB")

if image:
    img_array = np.array(image)

    with st.spinner("Analysing sign..."):
        cropped, boxes, dino_classes = detect_signs(img_array)

        # ── Fallback: color detector when DINO finds nothing ──────────────
        used_fallback = False
        if not boxes:
            boxes, dino_classes = detect_signs_by_color(img_array)
            cropped = None
            used_fallback = bool(boxes)
            if used_fallback:
                print("[main] DINO found nothing — using color detector fallback")

        # ── Build per-sign list sorted bottom-to-top ──────────────────────
        signs = []
        if boxes:
            paired = sorted(
                zip(boxes, dino_classes),
                key=lambda bc: bc[0][3],   # sort by y2, descending = bottom first
                reverse=True,
            )
            for box, dino_label in paired:
                x1, y1, x2, y2 = box
                sign_crop = img_array[y1:y2, x1:x2]
                text       = extract_text(sign_crop) if sign_crop.size > 0 else ""
                symbol_cls = classify_symbol(sign_crop) if not text.strip() else None
                signs.append({
                    "text":       text,
                    "dino_label": dino_label,
                    "symbol_cls": symbol_cls,
                })

        # ── Classify: OCR text first, DINO label as fallback ──────────────
        signs = classify_signs(signs)

        now    = datetime.now()
        result = parse_parking_rules(signs, now)

    # ── Display ───────────────────────────────────────────────────────────
    if dev_mode:
        st.subheader("Detection preview")
        if boxes:
            caption = f"{len(boxes)} region(s) detected"
            if used_fallback:
                caption += " (color detector fallback)"
            st.image(draw_boxes(img_array, boxes), caption=caption, width="stretch")
        else:
            st.image(image, caption="No detections", width="stretch")
    else:
        st.image(image, caption="Input image", width="stretch")

    if not signs:
        st.warning("No parking sign detected. Try a clearer photo.")
    else:
        if result["can_park"] is True:
            st.success(f"✅ You can park here.  \n{result['message']}")
        elif result["can_park"] is False:
            st.error(f"🚫 You cannot park here.  \n{result['message']}")
        else:
            st.warning(f"⚠️ Could not determine.  \n{result['message']}")

        for note in result.get("notes", []):
            st.info(f"ℹ️ {note}")

        if dev_mode:
            st.divider()
            st.subheader("Developer info")
            st.caption("**Per-sign classification (bottom → top):**")
            for s in signs:
                sym = s.get('symbol_cls') or ''
                st.caption(
                    f"`{s['class']}` | "
                    f"OCR: `{s.get('text','')[:60]}` | "
                    f"DINO: `{s.get('dino_label','')[:60]}`"
                    + (f" | CLIP: `{sym}`" if sym else "")
                )
            st.divider()
            st.caption("**Anchor groups:**")
            for g in result.get("groups", []):
                ovrig = " [övrig tid]" if g.get("ovrig_tid") else ""
                st.caption(f"`{g['anchor']}`{ovrig} — intervals: {g.get('intervals')} — mod: `{g.get('mod_text','')[:80]}`")
            st.divider()
            st.json(result)