import streamlit as st
from PIL import Image
import numpy as np
from detector import detect_sign, draw_boxes
from ocr import extract_text
from parser import parse_parking_rules
from datetime import datetime

st.set_page_config(
    page_title="ParkSence",
    page_icon="🅿️",
    layout="centered",
)

# Sidebar
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
        cropped, boxes, detected_classes = detect_sign(img_array)

        if dev_mode:
            st.subheader("Detection preview")
            if boxes:
                annotated = draw_boxes(img_array, boxes)
                st.image(annotated, caption=f"{len(boxes)} sign(s) detected", width="stretch")
            else:
                st.image(image, caption="No boxes detected", width="stretch")
        else:
            st.image(image, caption="Input image", width="stretch")

        if cropped is None and not detected_classes:
            st.warning("No parking sign detected. Try a clearer photo.")
        else:
            text = extract_text(cropped) if cropped is not None else ""

            now = datetime.now()
            result = parse_parking_rules(text, now, list(set(detected_classes)))

            if result["can_park"]:
                st.success(f"✅ You can park here.  \n{result['message']}")
            else:
                st.error(f"🚫 You cannot park here.  \n{result['message']}")

            if result.get("notes"):
                for note in result["notes"]:
                    st.info(f"ℹ️ {note}")

            if dev_mode:
                st.divider()
                st.subheader("Developer info")
                st.caption(f"Raw OCR text: `{text}`")
                st.caption(f"Detected classes: {detected_classes}")
                st.json(result)
