import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
from datetime import datetime

from torchgen import model

# Make project root importable when running from /app
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LOG_FILE = ROOT_DIR / "logs" / "predictions.log"

from src.predict import (
    get_device,
    load_class_mapping,
    load_model,
    get_inference_transform,
    predict_topk,
)

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

def get_model_version(weights_path: Path) -> str:
    return weights_path.stem

@st.cache_resource
def load_artifacts():
    weights_path = ROOT_DIR / "artifacts" / "models" / "model_v1.pt"
    class_map_path = ROOT_DIR / "artifacts" / "reports" / "class_to_idx.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"Missing class map: {class_map_path}")

    device = get_device()
    class_to_idx, idx_to_class = load_class_mapping(str(class_map_path))
    model = load_model(
        weights_path=str(weights_path),
        num_classes=len(class_to_idx),
        device=device,
    )
    transform = get_inference_transform(img_size=224)

    model_version = get_model_version(weights_path)
    return model, idx_to_class, transform, device, model_version    


def main():
    st.title("Plant Disease Classifier")
    st.write("Upload a plant leaf image to predict the disease class.")

    try:
        model, idx_to_class, transform, device, model_version = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Running on: {device} | Model: {model_version}")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Running inference..."):
                image_tensor = transform(image).unsqueeze(0)
                image_tensor = image_tensor.to(device)

                predictions = predict_topk(
                    model=model,
                    image_tensor=image_tensor,
                    idx_to_class=idx_to_class,
                    device=device,
                    top_k=5,
                )

            st.subheader("Top Prediction")
            top_label, top_prob = predictions[0]
            status = "LOW_CONFIDENCE" if top_prob < 0.60 else "OK"
            st.success(f"{top_label} ({top_prob:.4f})")

            if status == "LOW_CONFIDENCE":
                st.warning("Low-confidence prediction. The model may be uncertain about this image.")
            
            os.makedirs(LOG_FILE.parent, exist_ok=True)

            if not LOG_FILE.exists():
                with open(LOG_FILE, "w") as f:
                    f.write("timestamp | model_version | image_name | predicted_class | confidence | status\n")

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{model_version} | "
                    f"{image_name} | "
                    f"{top_label} | "
                    f"{top_prob:.4f} | "
                    f"{status}\n"
                )

            st.subheader("Top 5 Predictions")
            for label, prob in predictions:
                st.write(f"**{label}** — {prob:.4f}")
                st.progress(float(prob))


if __name__ == "__main__":
    main()