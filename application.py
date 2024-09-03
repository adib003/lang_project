import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# Parameters
MODEL_PATH = 'sign_language_model.pth'
CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
IMG_SIZE = 64
ROI_SIZE = 200  # Region of Interest size for capturing hand signs

# Define the model (assuming the same model structure used in training)
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjusted the input size based on image size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 16 * 16)  # Adjusted size based on image size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

sentence = ""
current_word = ""
print("Start showing signs. Press 'Space' for next word, 'Enter' for next sentence, 'Esc' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and resize the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Define ROI (Region of Interest)
    height, width = frame.shape[:2]
    start_x = (width - ROI_SIZE) // 2
    start_y = (height - ROI_SIZE) // 2
    end_x = start_x + ROI_SIZE
    end_y = start_y + ROI_SIZE

    # Draw the ROI on the frame
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Display countdown for next sign capture
    for i in range(3, 0, -1):
        temp_frame = frame.copy()

        # Overlay countdown timer
        cv2.putText(temp_frame, f"Capturing in {i}...", (start_x, start_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay instructions
        cv2.putText(temp_frame, "Press 'Space' for next word, 'Enter' for next sentence, 'Esc' to exit.",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display frame with overlays
        cv2.imshow('Sign Language Recognition', temp_frame)
        cv2.waitKey(1000)  # Wait 1 second for countdown

    # Preprocess the ROI for prediction
    roi = frame[start_y:end_y, start_x:end_x]
    roi_transformed = transform(roi)
    roi_transformed = roi_transformed.unsqueeze(0).to(device)

    # Predict the gesture
    with torch.no_grad():
        outputs = model(roi_transformed)
        _, predicted = torch.max(outputs, 1)
        predicted_gesture = CLASSES[predicted.item()]

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted: {predicted_gesture}", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    key = cv2.waitKey(1000) & 0xFF  # Wait for key press or move to next sign automatically

    if key == 32:  # Space key for next word
        sentence += current_word + " "
        current_word = ""
        print(f"Current Sentence: {sentence.strip()}")
    elif key == 13:  # Enter key for next sentence
        sentence += current_word
        print(f"Final Sentence: {sentence.strip()}")
        sentence = ""
        current_word = ""
    elif key == 27:  # Esc key to exit
        print("Exiting application.")
        break
    else:
        current_word += predicted_gesture

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
