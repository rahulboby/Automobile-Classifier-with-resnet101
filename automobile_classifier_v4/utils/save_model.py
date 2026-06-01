#Function to save the model
import os

from sklearn import utils
import torch

from app import CLASS_NAMES, NUM_CLASSES


def save_model(model, folder_name, model_name):
    os.makedirs(folder_name, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "image_size": utils.configs.IMAGE_SIZE,
        "architecture": "ResNet50"
    }, f"{folder_name}/{model_name}.pth")

    print(f"Model saved to {folder_name}/{model_name}.pth")