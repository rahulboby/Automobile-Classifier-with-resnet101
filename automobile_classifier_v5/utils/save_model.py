import os
from pathlib import Path

import torch

from utils.configs import IMAGE_SIZE, PROJECT_ROOT


def infer_architecture(model_name):
    model_name = Path(model_name).name
    if model_name.startswith("final_model_"):
        return model_name.replace("final_model_", "").replace(".pth", "")
    return model_name


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_model(model, folder_name, model_name, class_names, num_classes, image_size=IMAGE_SIZE, architecture=None):
    folder_path = resolve_project_path(folder_name)
    os.makedirs(folder_path, exist_ok=True)
    if architecture is None:
        architecture = infer_architecture(model_name)
    model_path = folder_path / model_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": image_size,
        "architecture": architecture
    }, model_path)

    print(f"Model saved to {model_path}")
