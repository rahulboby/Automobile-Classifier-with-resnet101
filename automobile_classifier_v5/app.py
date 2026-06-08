import csv
from pathlib import Path
import time

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.inference import load_model, predict, transform_image
import utils.configs
from utils.configs import DEVICE
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="KIA Model Classifier (Testing)",
    layout="wide"
)

# -----------------------------
# LOAD ALL MODELS IN models/ FOR TESTING
# -----------------------------
MODEL_DIR = Path(utils.configs.MODEL_DIR)

def load_all_models(models_dir=MODEL_DIR):
    models_info = []
    if not models_dir.exists():
        return models_info

    pths = sorted(models_dir.glob("*.pth"))
    for p in pths:
        try:
            m, class_names, num_classes, image_size, architecture = load_model(str(p))
            models_info.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "model": m,
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "architecture": architecture,
                }
            )
        except Exception as e:
            # non-fatal: skip corrupted or incompatible checkpoints
            print(f"Skipping model {p}: {e}")

    return models_info


ALL_MODELS = load_all_models()


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


def get_model_selection():
    model_names = [m["filename"] for m in ALL_MODELS]
    if not model_names:
        return []
    selected = st.multiselect(
        "Select models to evaluate",
        options=model_names,
        default=model_names,
        help="Choose which model checkpoint files to include in batch evaluation.",
    )
    return [m for m in ALL_MODELS if m["filename"] in selected]


def build_summary_dataframe(results_df):
    if results_df.empty:
        return results_df

    summary = results_df.groupby(["model_file", "architecture"]).agg(
        images_tested=("image_filename", "nunique"),
        image_predictions=("image_filename", "count"),
        avg_top1_conf=("top1_conf", "mean"),
        avg_top2_conf=("top2_conf", "mean"),
        avg_top3_conf=("top3_conf", "mean"),
        most_common_top1=("top1", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        top1_mode_count=("top1", lambda x: x.value_counts().iloc[0] if not x.empty else 0),
    ).reset_index()

    summary["avg_top1_conf"] = summary["avg_top1_conf"].round(4)
    summary["avg_top2_conf"] = summary["avg_top2_conf"].round(4)
    summary["avg_top3_conf"] = summary["avg_top3_conf"].round(4)
    summary["top1_mode_pct"] = (summary["top1_mode_count"] / summary["images_tested"]).round(4)
    summary = summary.sort_values(by=["avg_top1_conf", "top1_mode_pct"], ascending=False)
    return summary


# -----------------------------
# MAIN PAGE
# -----------------------------
if st.session_state.page == "main":

    st.title("KIA Model Classifier")
    st.write("Upload one or more images for classification. The app will run them through every selected model.")

    selected_models = get_model_selection()

    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if not ALL_MODELS:
            st.error("No model checkpoints found in the models/ directory.")
        elif not selected_models:
            st.warning("Select at least one model to evaluate.")
        else:
            batch_results = []
            total_start = time.perf_counter()

            for uploaded_file in uploaded_files:
                image_name = getattr(uploaded_file, "name", "uploaded_image")
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception as e:
                    st.warning(f"Unable to open {image_name}: {e}")
                    continue

                for minfo in selected_models:
                    model_path = minfo["path"]
                    filename = minfo["filename"]
                    model_obj = minfo["model"]
                    class_names = minfo["class_names"]
                    image_size = minfo["image_size"]
                    architecture = minfo.get("architecture", "")

                    transform = lambda img, s=image_size: transform_image(img, s)

                    model_start = time.perf_counter()
                    try:
                        _, top_predictions = predict(
                            image=image,
                            model=model_obj,
                            transform=transform,
                            class_names=class_names,
                            top_k=3,
                        )
                    except Exception as e:
                        st.warning(f"Prediction failed for {filename} on {image_name}: {e}")
                        continue
                    inference_time = time.perf_counter() - model_start

                    row = {
                        "image_filename": image_name,
                        "model_file": filename,
                        "architecture": architecture,
                        "top1": top_predictions[0]["class_name"] if len(top_predictions) > 0 else "",
                        "top1_conf": float(top_predictions[0]["confidence"]) if len(top_predictions) > 0 else 0.0,
                        "top2": top_predictions[1]["class_name"] if len(top_predictions) > 1 else "",
                        "top2_conf": float(top_predictions[1]["confidence"]) if len(top_predictions) > 1 else 0.0,
                        "top3": top_predictions[2]["class_name"] if len(top_predictions) > 2 else "",
                        "top3_conf": float(top_predictions[2]["confidence"]) if len(top_predictions) > 2 else 0.0,
                        "inference_time_sec": float(f"{inference_time:.4f}"),
                    }

                    batch_results.append(row)

                    append_prediction_log(
                        uploaded_filename=image_name,
                        model_path=model_path,
                        device=DEVICE,
                        architecture=architecture,
                        top_predictions=top_predictions,
                        inference_time_sec=inference_time,
                    )

            total_time = time.perf_counter() - total_start

            if batch_results:
                df = pd.DataFrame(batch_results)
                st.subheader("Model predictions")
                st.dataframe(df)

                summary_df = build_summary_dataframe(df)
                st.subheader("Batch model ranking")
                st.dataframe(summary_df)

                best_model = summary_df.iloc[0]
                st.markdown(
                    f"**Best model by average Top-1 confidence:** {best_model['model_file']} ({best_model['architecture']}) — {best_model['avg_top1_conf']:.4f}%"
                )
                st.markdown(
                    f"Most consistent Top-1 label for this model: {best_model['most_common_top1']} ({best_model['top1_mode_count']} / {best_model['images_tested']} images)"
                )
                st.text(f"Total inference time across all models and images: {total_time:.3f} sec")
            else:
                st.write("No successful predictions to display.")

            if len(uploaded_files) == 1:
                with st.expander("Show Uploaded Image", expanded=False):
                    st.image(image, caption=image_name, width=500)
            else:
                st.write(f"Processed {len(uploaded_files)} uploaded images.")


    st.divider()

    if st.button("View Model Details"):
        go_details()
        st.rerun()
    # print("DEBUG: MODEL details rendered successfully.")

    with st.expander("View Prediction Log", expanded=False):
        log_file = get_prediction_log_path()
        if log_file.exists():
            df = pd.read_csv(log_file)
            st.dataframe(df)
        else:
            st.write("No predictions logged yet.")

# -----------------------------
# DETAILS PAGE
# -----------------------------
elif st.session_state.page == "details":

    st.title("Loaded Models")

    st.write(f"Models discovered in: {MODEL_DIR}")
    st.write(f"Number of model checkpoints loaded: {len(ALL_MODELS)}")

    for minfo in ALL_MODELS:
        st.subheader(minfo.get("filename", "unknown"))
        st.write(f"Architecture: {minfo.get('architecture', '')}")
        st.write(f"Classes: {len(minfo.get('class_names', []))}")
        st.write(f"Image size expected: {minfo.get('image_size', '')}")
        if st.button(f"Show classes for {minfo.get('filename')}", key=minfo.get('filename')):
            st.write(minfo.get('class_names', []))

    st.subheader("Inference Device")
    st.write(str(DEVICE).upper())

    st.divider()

    if st.button("Back to Classifier"):
        go_main()
        st.rerun()