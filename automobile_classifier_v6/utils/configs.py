from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

# Paths
DATA_DIR = project_path("data")
SPLIT_DIR = project_path("data_split")
MODEL_DIR = project_path("models")

def get_checkpoint_dir(model_name):
    return MODEL_DIR / f"checkpoints_{model_name}"


def get_model_filename(model_name):
    return f"final_model_{model_name}.pth"


def get_model_path(model_name):
    return MODEL_DIR / get_model_filename(model_name)

# Image settings
IMAGE_SIZE = 512

# Training settings
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 4e-6
CHECKPOINT_INTERVAL = 5  # Save a checkpoint every 5 epochs

# Dataset split ratios
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

MODEL_NAME = "densenet121"  # Possible values: resnet50, resnet18, resnet101, efficientnet_b0, densenet121, mobilenet_v2
MODEL_PATH = get_model_path(MODEL_NAME)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
