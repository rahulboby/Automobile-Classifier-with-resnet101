import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/final_model_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint["image_size"]
    architecture = checkpoint["architecture"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model, class_names, num_classes, image_size, architecture


model, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, ARCHITECTURE = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, prediction = torch.max(probabilities, dim=1)

        predicted_class = CLASS_NAMES[prediction.item()]
        confidence_score = confidence.item() * 100

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence_score:.2f}%")

    st.divider()

    if st.button("View Model Details"):
        go_details()
        st.rerun()

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