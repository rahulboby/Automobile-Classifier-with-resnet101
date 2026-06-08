# gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_gradcam(model, image_tensor, original_image, target_class):
    """
    Custom Grad-CAM implementation for ResNet50
    No external dependencies except PyTorch + numpy + OpenCV
    """
    model.eval()
    
    # We'll hook into the last convolutional layer (layer4[-1])
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # Register hooks
    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)       

    # Forward pass
    image_tensor = image_tensor.requires_grad_(True)
    output = model(image_tensor)
    
    # Backward pass for target class
    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Get activations and gradients
    act = activations[0]           # shape: (C, H, W)
    grad = gradients[0]            # shape: (C, H, W)

    # Global Average Pooling on gradients
    weights = torch.mean(grad, dim=(1, 2), keepdim=True)  # shape: (C, 1, 1)

    # Weighted combination
    cam = torch.sum(weights * act, dim=0)  # shape: (H, W)

    # ReLU and normalize
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # Convert to numpy
    cam = cam.cpu().numpy()

    # Resize CAM to original image size
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    original_image = (original_image * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return overlay