from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models
from configs import DEVICE

def load_model(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

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

def predict(image, model, transform, class_names):

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = class_names[prediction.item()]
    confidence_score = confidence.item() * 100

    return (
        predicted_class,
        confidence_score,
        image_tensor,
        prediction.item()
    )

def transform_train_image(image, image_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform(image)

def transform_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image)