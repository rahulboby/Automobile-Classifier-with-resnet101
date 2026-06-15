import csv
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.inference import load_model, predict, transform_image
from utils.configs import DEVICE, MODEL_PATH
import numpy as np
from utils.gradcam import generate_gradcam
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="KIA Model Classifier",
    layout="wide"
)

# -----------------------------
# MODEL LOADING
# -----------------------------
model, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, ARCHITECTURE = load_model(MODEL_PATH)
# print("DEBUG: Model loaded successfully.")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = lambda image: transform_image(image, IMAGE_SIZE)
# print("DEBUG: Transform function defined successfully.")

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"

# -----------------------------
# PAGE FUNCTIONS
# -----------------------------
def go_main():
    st.session_state.page = "main"


def go_details():
    st.session_state.page = "details"


def get_prediction_log_path():
    current_dir = Path(__file__).resolve().parent
    log_dir = current_dir.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "prediction_log.csv"


def append_prediction_log(uploaded_filename, model_path, device, architecture, top_predictions, inference_time_sec):
    log_file = get_prediction_log_path()
    fieldnames = [
        "timestamp",
        "image_filename",
        "model_path",
        "device",
        "architecture",
        "predicted_class",
        "predicted_confidence",
        "second_class",
        "second_confidence",
        "third_class",
        "third_confidence",
        "inference_time_sec"
    ]

    record = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "image_filename": uploaded_filename,
        "model_path": model_path,
        "device": str(device),
        "architecture": architecture,
        "predicted_class": top_predictions[0]["class_name"],
        "predicted_confidence": f"{top_predictions[0]['confidence']:.4f}",
        "second_class": top_predictions[1]["class_name"] if len(top_predictions) > 1 else "",
        "second_confidence": f"{top_predictions[1]['confidence']:.4f}" if len(top_predictions) > 1 else "",
        "third_class": top_predictions[2]["class_name"] if len(top_predictions) > 2 else "",
        "third_confidence": f"{top_predictions[2]['confidence']:.4f}" if len(top_predictions) > 2 else "",
        "inference_time_sec": f"{inference_time_sec:.4f}"
    }

    file_exists = log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    return log_file

# -----------------------------
# MAIN PAGE
# -----------------------------
if st.session_state.page == "main":

    st.title("KIA Model Classifier")
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        
        start_time = datetime.now()

        image = Image.open(uploaded_file).convert("RGB")

        image_tensor, top_predictions = predict(
            image=image,
            model=model,
            transform=transform,
            class_names=CLASS_NAMES,
            top_k=3
        )

        predicted_class = top_predictions[0]["class_name"]
        confidence_score = top_predictions[0]["confidence"]
        predicted_index = top_predictions[0]["index"]

        original_image = np.array(image).astype(np.float32) / 255.0

        with torch.enable_grad():
            gradcam_image = generate_gradcam(
                model=model,
                image_tensor=image_tensor,
                original_image=original_image,
                target_class=predicted_index
            )
        

        st.success(f"Top Prediction: {predicted_class.upper()}")
        st.progress(int(confidence_score), text=f"Confidence: {confidence_score:.2f}%")

        with st.expander("More details", expanded=False):
            end_time = datetime.now()
            inference_time_min = (end_time - start_time).total_seconds()
            inference_time_min = max(inference_time_min, 0.001)  # Ensure minimum time for display
            st.text(f"Inference Time: {inference_time_min:.2f} seconds")
            st.subheader("Other Top Predictions")
            for rank, prediction in enumerate(top_predictions[1:], start=2):
                class_name = prediction["class_name"]
                confidence = prediction["confidence"]
                st.write(f"{rank}. {class_name} — {confidence:.2f}%")

        

        append_prediction_log(
            uploaded_filename=getattr(uploaded_file, "name", "uploaded_image"),
            model_path=MODEL_PATH,
            device=DEVICE,
            architecture=ARCHITECTURE,
            top_predictions=top_predictions,
            inference_time_sec=inference_time_min
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Grad-CAM Visualization", expanded=False):
                st.write("Grad-CAM visualization is currently unavailable.")
                st.image(
                    gradcam_image,
                    caption="Grad-CAM (Model Attention) Heatmap",
                    width=500
                )
        with col2:
            with st.expander("Show Uploaded Image", expanded=False):
                st.image(
                    image,
                    caption="Uploaded Image",
                    width=500
                )


    st.divider()

    if st.button("View Model Details"):
        go_details()
        st.rerun()
    # print("DEBUG: MODEL details rendered successfully.")

    with st.expander("View Prediction Log", expanded=False):
        log_file = get_prediction_log_path()
        if log_file.exists():
            import pandas as pd
            df = pd.read_csv(log_file)
            st.dataframe(df)
        else:
            st.write("No predictions logged yet.")

# -----------------------------
# DETAILS PAGE
# -----------------------------
elif st.session_state.page == "details":

    st.title("Model Details")

    st.subheader("Architecture")
    st.write(ARCHITECTURE)

    st.subheader("Classifier Head")
    st.code(f"Linear(2048 → {NUM_CLASSES})")

    st.subheader("Detected Classes ({NUM_CLASSES})")
    st.write(CLASS_NAMES)

    st.subheader("Input Specifications")
    st.write(f"Image Size: {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write("Input Type: RGB Image")

    # st.subheader("Preprocessing")
    # st.code("""
    #     Resize()
    #     ToTensor()
    #     Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # """)

    st.subheader("Training Setup")
    st.write("Transfer Learning")
    st.write("Pretrained ImageNet ResNet50 Backbone - Unfrozen Layer 4 and Classifier Head")
    st.write("Custom Final Classification Layer")
    st.write("Loss Function: CrossEntropyLoss")
    st.write("Optimizer: Adam")

    st.subheader("Model File")
    st.code(MODEL_PATH)

    st.subheader("Inference Device")
    st.write(str(DEVICE).upper())

    st.divider()

    if st.button("Back to Classifier"):
        go_main()
        st.rerun()