import torch
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import timm
from torch import nn

# Use constants from your training script
MODEL_NAME = "densenet121"
IMG_SIZE = [512, 512]
IN_CHANS = 30
N_CLASSES = 75
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/rsna24-results"
# Path to the best model
fname = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/rsna24-results/best_accuracy_model_fold-1.pt"

# Conditions and levels (from your training script)
CONDITIONS = [
    'Spinal Canal Stenosis',
    'Left Neural Foraminal Narrowing',
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1'
]

# Define the DenseNet model (same as in your training script)


class DenseNet(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )

    def forward(self, x):
        return self.model(x)


# Preprocessing: Albumentations transforms for validation
transforms_val = Compose([
    Resize(IMG_SIZE[0], IMG_SIZE[1]),
    Normalize(mean=0.5, std=0.5),
    ToTensorV2()  # Convert to PyTorch tensor
])

# Load the model weights


def load_model(weights_path):
    model = DenseNet(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess a single image


# Preprocess an image object
def preprocess_image(image):
    # Convert the image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.array(image)

    # Repeat the 3 channels to match 30 channels
    img = np.repeat(img[:, :, :, np.newaxis], 10, axis=-1)
    img = img.reshape(img.shape[0], img.shape[1], 30)  # Reshape to (H, W, 30)

    # Apply validation transforms
    transformed = transforms_val(image=img)
    input_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
    return input_tensor.to(DEVICE)


# Perform inference


def classify_image(image):
    model = load_model(fname)
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        logits = model(input_tensor)  # Raw model outputs
        predictions = logits.softmax(dim=1)  # Convert logits to probabilities

    # Map predictions to conditions and levels
    id2label = {0: 'Normal/Mild', 1: 'Moderate', 2: 'Severe'}
    results = {}
    for i, condition in enumerate(CONDITIONS):
        for j, level in enumerate(LEVELS):
            col = i * len(LEVELS) + j
            sub_logits = predictions[:, col * 3: col * 3 + 3]
            sub_pred = sub_logits.argmax(dim=1).item()
            results[f'{condition} at {level}'] = id2label[sub_pred]

    # Filter out anything labeled 'Normal/Mild'
    results = {k: v for k, v in results.items() if v != 'Normal/Mild'}

    return results


# Main function for inference
if __name__ == "__main__":
    # Path to the saved model weights
    weights_path = fname  # Use the fname variable for the model

    # Path to the input image
    # Update with the actual image path
    img_path = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/cvt_png/4003253/Axial T2/001.png"

    # Load the model
    print("Loading model...")
    model = load_model(weights_path)

    # Preprocess the image

    # Perform classification
    print("Classifying image...")
    results = classify_image(model, img_path)

    # Print the results
    print("Classification Results:")
    for key, prediction in results.items():
        if (prediction == "Severe" or prediction == "Moderate"):
            print(f"{key}: {prediction}")
