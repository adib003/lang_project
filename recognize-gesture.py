import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = 'sign_language_model.pth'
CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
IMG_SIZE = 64  # Use the same size as in training
ROI_SIZE = 400  # Adjust as needed
FRAME_WIDTH = 640  # Reduce the resolution for faster processing
FRAME_HEIGHT = 480  # Reduce the resolution for faster processing
CAPTURE_INTERVAL = 3  # Capture interval in seconds

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=36):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (IMG_SIZE // 4) * (IMG_SIZE // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(num_classes=len(CLASSES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except RuntimeError as e:
    print(f"Error loading model state_dict: {e}")
    raise
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

text = ""
last_prediction = ""
start_time = cv2.getTickCount()
captured_text = ""
print("Instructions:\nPress 'Space' to add a space, 'Enter' to end the sentence, 'q' to start a new sentence, 'Esc' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    start_x = (width - ROI_SIZE) // 2
    start_y = (height - ROI_SIZE) // 2
    end_x = start_x + ROI_SIZE
    end_y = start_y + ROI_SIZE

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Get the region of interest and transform it
    roi = frame[start_y:end_y, start_x:end_x]
    roi_pil = Image.fromarray(roi)
    roi_transformed = transform(roi_pil)
    roi_transformed = roi_transformed.unsqueeze(0).to(device)

    # Predict gesture
    with torch.no_grad():
        outputs = model(roi_transformed)
        _, predicted = torch.max(outputs, 1)
        predicted_gesture = CLASSES[predicted.item()]

    # Update text and check if it's time to capture a new prediction
    current_time = cv2.getTickCount()
    if (current_time - start_time) / cv2.getTickFrequency() >= CAPTURE_INTERVAL:
        if last_prediction:
            captured_text += last_prediction
        last_prediction = predicted_gesture
        start_time = current_time

    cv2.putText(frame, f"Predicted: {predicted_gesture}", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display text and instructions on the frame
    display_text = f"Text: {captured_text} {last_prediction}"
    cv2.putText(frame, display_text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'Space' to add a space, 'Enter' to end the sentence, 'q' to start a new sentence, 'Esc' to stop.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Spacebar
        captured_text += " "
    elif key == 13:  # Enter
        print(f"Final Sentence: {captured_text.strip()}")
        captured_text = ""
    elif key == ord('q'):  # 'q' key
        captured_text += " "  # Add space before starting new sentence
        print(f"Starting a new sentence. Current text: {captured_text.strip()}")
    elif key == 27:  # Esc
        print("Exiting application.")
        break

cap.release()
cv2.destroyAllWindows()
