import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms as T
import numpy as np

# Parameters
MODEL_PATH = 'sign_language_model.pth'
IMG_SIZE = 256
DATA_DIR = 'sign_language_data'

# Define the CNN model
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 64 * 64)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
def load_model(model_path, num_classes):
    model = SignLanguageModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Predict function
def predict_image(model, image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Main function
def main():
    # Load class names
    classes = sorted(os.listdir(DATA_DIR))

    # Load model
    num_classes = len(classes)
    model = load_model(MODEL_PATH, num_classes)

    # Test with a sample image
    test_image_path = 'test_image.png'  # Replace with your image path
    predicted_class = predict_image(model, test_image_path)

    # Display result
    print(f'Predicted class: {classes[predicted_class]}')

if __name__ == "__main__":
    main()
