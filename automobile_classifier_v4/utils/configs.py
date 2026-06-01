import torch

# Paths
DATA_DIR = "data"
SPLIT_DIR = "data_split"

# Image settings
IMAGE_SIZE = 512

# Training settings
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 4e-6

# Dataset split ratios
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

MODEL_PATH = "models/final_model_v3.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")