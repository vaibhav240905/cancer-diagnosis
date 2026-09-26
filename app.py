import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICT_SCRIPT = PROJECT_ROOT / "predict.py"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def get_latest_image(pattern_suffix: str):
    if not FIGURES_DIR.exists():
        return None
    candidates = [
        p for p in FIGURES_DIR.glob(f"*{pattern_suffix}.png") if p.is_file()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_predictions(log_text: str):
    parsed = {"roi": None, "full": None}
    roi_match = re.search(r"ROI prediction \| label=(\w+) prob=([0-9.]+)", log_text, re.IGNORECASE)
    full_match = re.search(r"Full prediction \| label=(\w+) prob=([0-9.]+)", log_text, re.IGNORECASE)

    if roi_match:
        parsed["roi"] = {"label": roi_match.group(1).lower(), "prob_malignant": float(roi_match.group(2))}
    if full_match:
        parsed["full"] = {"label": full_match.group(1).lower(), "prob_malignant": float(full_match.group(2))}
    return parsed


def predicted_confidence(label: str, prob_malignant: float) -> float:
    if label == "malignant":
        return prob_malignant * 100.0
    return (1.0 - prob_malignant) * 100.0


def render_prediction_text(label: str, prob_malignant: float):
    is_malignant = label == "malignant"
    color = "#d62728" if is_malignant else "#2e8b57"  # red / green
    pretty_label = "MALIGNANT" if is_malignant else "BENIGN"
    conf = predicted_confidence(label, prob_malignant)
    st.markdown(
        f"""
        <div style="padding:10px 0 8px 0;">
          <div style="font-size:30px; font-weight:800; color:{color};">
            Prediction: {pretty_label}
          </div>
          <div style="font-size:20px; font-weight:600; color:#222;">
            Confidence: {conf:.2f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def split_roi_panel(panel_path: Path):
    # ROI panel from predict.py has 4 columns:
    # [Original | Predicted Mask | ROI Crop | Grad-CAM]
    img = Image.open(panel_path).convert("RGB")
    w, h = img.size
    sw = w // 4
    original = img.crop((0, 0, sw, h))
    seg_preview = img.crop((sw, 0, 2 * sw, h))
    gradcam = img.crop((3 * sw, 0, w, h))
    return original, seg_preview, gradcam


def split_full_panel(panel_path: Path):
    # Full panel from predict.py has 2 columns:
    # [Original | Grad-CAM]
    img = Image.open(panel_path).convert("RGB")
    w, h = img.size
    sw = w // 2
    original = img.crop((0, 0, sw, h))
    gradcam = img.crop((sw, 0, w, h))
    return original, gradcam


def run_prediction(image_path: str, mode: str):
    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--image_path",
        image_path,
        "--mode",
        mode,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return completed


def main():
    st.set_page_config(page_title="Skin Cancer Detection", layout="wide")
    st.title("Skin Cancer Detection")

    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox("Prediction Mode", ["roi", "full", "both"], index=0)

    uploaded_file = st.file_uploader(
        "Upload dermoscopic image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        preview = Image.open(uploaded_file).convert("RGB")
        st.image(preview, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction", type="primary"):
        if uploaded_file is None:
            st.error("Please upload an image first.")
            return

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_image_path = tmp.name

        try:
            with st.spinner("Running inference..."):
                result = run_prediction(temp_image_path, mode)

            logs = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode != 0:
                st.error("Prediction failed. See details below:")
                st.code(logs)
                return

            st.success("Prediction completed successfully.")
            parsed = parse_predictions(logs)

            roi_img_path = get_latest_image("_prediction_roi")
            full_img_path = get_latest_image("_prediction_full")

            if mode == "roi":
                st.subheader("ROI Prediction")
                if roi_img_path is not None:
                    roi_pred = parsed.get("roi")
                    if roi_pred is not None:
                        render_prediction_text(roi_pred["label"], roi_pred["prob_malignant"])
                    original, seg_preview, gradcam = split_roi_panel(roi_img_path)
                    st.markdown("**Original Image**")
                    st.image(original, use_container_width=True)
                    st.markdown("**Grad-CAM**")
                    st.image(gradcam, use_container_width=True)
                    st.markdown("**Segmentation Preview (small)**")
                    st.image(seg_preview, width=220)
                else:
                    st.warning("ROI prediction image not found in outputs/figures.")
            elif mode == "full":
                st.subheader("Full Image Prediction")
                if full_img_path is not None:
                    full_pred = parsed.get("full")
                    if full_pred is not None:
                        render_prediction_text(full_pred["label"], full_pred["prob_malignant"])
                    original, gradcam = split_full_panel(full_img_path)
                    st.markdown("**Original Image**")
                    st.image(original, use_container_width=True)
                    st.markdown("**Grad-CAM**")
                    st.image(gradcam, use_container_width=True)
                else:
                    st.warning("Full prediction image not found in outputs/figures.")
            else:
                st.subheader("ROI vs Full Image Comparison")
                st.write("")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### ROI Output")
                    if roi_img_path is not None:
                        roi_pred = parsed.get("roi")
                        if roi_pred is not None:
                            render_prediction_text(roi_pred["label"], roi_pred["prob_malignant"])
                        original, seg_preview, gradcam = split_roi_panel(roi_img_path)
                        st.markdown("**Original Image**")
                        st.image(original, use_container_width=True)
                        st.markdown("**Grad-CAM**")
                        st.image(gradcam, use_container_width=True)
                        st.markdown("**Segmentation Preview (small)**")
                        st.image(seg_preview, width=200)
                    else:
                        st.warning("ROI prediction image not found.")
                with col2:
                    st.markdown("### Full Image Output")
                    if full_img_path is not None:
                        full_pred = parsed.get("full")
                        if full_pred is not None:
                            render_prediction_text(full_pred["label"], full_pred["prob_malignant"])
                        original, gradcam = split_full_panel(full_img_path)
                        st.markdown("**Original Image**")
                        st.image(original, use_container_width=True)
                        st.markdown("**Grad-CAM**")
                        st.image(gradcam, use_container_width=True)
                    else:
                        st.warning("Full prediction image not found.")

        finally:
            try:
                os.remove(temp_image_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
