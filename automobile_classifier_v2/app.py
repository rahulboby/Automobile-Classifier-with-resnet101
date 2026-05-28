import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from inference import load_model, predict, transform_image
from configs import DEVICE, MODEL_PATH
import numpy as np
from gradcam import generate_gradcam

# -----------------------------
# CONFIG
# -----------------------------

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚘",
    layout="centered"
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

# -----------------------------
# MAIN PAGE
# -----------------------------
if st.session_state.page == "main":

    st.title("Vehicle Image Classifier")
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        predicted_class, confidence_score, image_tensor, predicted_index = predict(
            image=image,
            model=model,
            transform=transform,
            class_names=CLASS_NAMES
        )
        print("DEBUG: Prediction completed successfully.")

        original_image = np.array(image).astype(np.float32) / 255.0

        with torch.enable_grad():
            gradcam_image = generate_gradcam(
                model=model,
                image_tensor=image_tensor,
                original_image=original_image,
                target_class=predicted_index
            )

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence_score:.2f}%")

        st.subheader("Grad-CAM Visualization")
        st.image(
            gradcam_image,
            caption="Grad-CAM (Model Attention) Heatmap",
            use_container_width=True
        )

    st.divider()

    if st.button("View Model Details"):
        go_details()
        st.rerun()
    # print("DEBUG: MODEL details rendered successfully.")

# -----------------------------
# DETAILS PAGE
# -----------------------------
elif st.session_state.page == "details":

    st.title("Model Details")

    st.subheader("Architecture")
    st.write(ARCHITECTURE)

    st.subheader("Classifier Head")
    st.code(f"Linear(2048 → {NUM_CLASSES})")

    st.subheader("Detected Classes")
    st.write(CLASS_NAMES)

    st.subheader("Number of Classes")
    st.write(NUM_CLASSES)

    st.subheader("Input Specifications")
    st.write(f"Image Size: {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write("Input Type: RGB Image")

    st.subheader("Preprocessing")
    st.code("""
        Resize()
        ToTensor()
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    """)

    st.subheader("Training Setup")
    st.write("Transfer Learning")
    st.write("Pretrained ImageNet ResNet50 Backbone")
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