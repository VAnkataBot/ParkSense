from pathlib import Path
from datetime import datetime
import streamlit as st
import numpy as np
from PIL import Image
from core.detector import detect_signs, draw_boxes
from core.ocr import extract_text
from core.classifier import classify_signs
from core.parser import parse_parking_rules
from core.symbol_classifier import classify_symbol

st.set_page_config(page_title="Batch Testing", page_icon="🧪", layout="wide")
st.title("🧪 Batch Testing")
st.caption("Run the full pipeline on a folder of images.")

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

folder_input = st.text_input("Folder path", placeholder="/path/to/images")
dev_mode = st.toggle("Show developer details", value=True)

if not folder_input:
    st.info("Enter a folder path to get started.")
    st.stop()

folder = Path(folder_input.strip())
if not folder.is_absolute():
    folder = Path(__file__).resolve().parent.parent / folder
if not folder.is_dir():
    st.error(f"Not a valid directory: `{folder}`")
    st.stop()

images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
if not images:
    st.warning("No supported images found in that folder.")
    st.stop()

st.write(f"Found **{len(images)}** image(s). Click Run to analyse.")
if not st.button("▶ Run Analysis"):
    st.stop()

now = datetime.now()
progress = st.progress(0, text="Starting…")

for i, img_path in enumerate(images):
    progress.progress((i + 1) / len(images), text=f"Processing {img_path.name} ({i+1}/{len(images)})")

    with st.container(border=True):
        col_img, col_result = st.columns([1, 2])

        try:
            image     = Image.open(img_path).convert("RGB")
            img_array = np.array(image)

            cropped, boxes, dino_labels = detect_signs(img_array)

            # Build per-sign list sorted bottom-to-top (y2 descending)
            signs = []
            if boxes:
                paired = sorted(
                    zip(boxes, dino_labels),
                    key=lambda bc: bc[0][3],
                    reverse=True,
                )
                for box, dino_label in paired:
                    x1, y1, x2, y2 = box
                    sign_crop = img_array[y1:y2, x1:x2]
                    text       = extract_text(sign_crop) if sign_crop.size > 0 else ""
                    symbol_cls = classify_symbol(sign_crop) if not text.strip() else None
                    signs.append({"text": text, "dino_label": dino_label, "symbol_cls": symbol_cls})

            # Classify: OCR text first, DINO label as fallback
            signs = classify_signs(signs)

            result = parse_parking_rules(signs, now)

            with col_img:
                st.caption(img_path.name)
                if dev_mode and boxes:
                    st.image(draw_boxes(img_array, boxes), width="stretch")
                else:
                    st.image(image, width="stretch")

            with col_result:
                if result["can_park"] is True:
                    st.success(f"✅ {result['message']}")
                elif result["can_park"] is False:
                    st.error(f"🚫 {result['message']}")
                else:
                    st.warning(f"⚠️ {result['message']}")

                for note in result.get("notes", []):
                    st.info(f"ℹ️ {note}")

                if dev_mode:
                    st.caption("**Per-sign classification (bottom → top):**")
                    for s in signs:
                        sym = s.get('symbol_cls') or ''
                        st.caption(
                            f"`{s['class']}` | "
                            f"OCR: `{s.get('text', '')[:60]}` | "
                            f"DINO: `{s.get('dino_label', '')[:60]}`"
                            + (f" | CLIP: `{sym}`" if sym else "")
                        )
                    st.caption("**Anchor groups:**")
                    for g in result.get("groups", []):
                        ovrig = " [övrig tid]" if g.get("ovrig_tid") else ""
                        st.caption(
                            f"`{g['anchor']}`{ovrig} — "
                            f"intervals: {g.get('intervals')} — "
                            f"mod: `{g.get('mod_text', '')[:80]}`"
                        )
                    st.json(result)

        except Exception as e:
            with col_img:
                st.caption(img_path.name)
            with col_result:
                st.error(f"Error: {e}")

progress.empty()
st.success("Done.")