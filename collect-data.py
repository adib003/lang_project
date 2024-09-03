import cv2
import os

# Parameters
DATA_DIR = 'sign_language_data'  # Directory to save data
CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')  # List of classes (gestures)
NUM_IMAGES = 100  # Number of images to capture per gesture
IMG_SIZE = 640  # Size of the main frame
ROI_SIZE = 300  # Size of the square ROI (Region of Interest)

# Create directory for each class
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)

# Start capturing images
cap = cv2.VideoCapture(0)
collecting = False
current_class = None
count = 0

print("Press a key corresponding to the class (0-9, A-Z) to start collecting data for that class.")
print("Press 'Esc' to quit the data collection.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Define ROI (Region of Interest) dimensions
    height, width = frame.shape[:2]
    start_x = (width - ROI_SIZE) // 2
    start_y = (height - ROI_SIZE) // 2
    end_x = start_x + ROI_SIZE
    end_y = start_y + ROI_SIZE

    # Draw the ROI on the frame
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    roi = frame[start_y:end_y, start_x:end_x]

    # Display the frame
    cv2.imshow('Frame', frame)

    # Display the ROI separately
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(1) & 0xFF

    # If a key is pressed and it's a valid class
    if key in [ord(c) for c in CLASSES]:
        current_class = chr(key).upper()  # Get the uppercase character from the key press
        count = 0
        collecting = True
        print(f"Started collecting images for class: {current_class}")

    # Start or continue collecting images for the current class
    if collecting and count < NUM_IMAGES:
        # Save the ROI as an image
        img_name = os.path.join(DATA_DIR, current_class, f'{current_class}_{count}.jpg')
        cv2.imwrite(img_name, roi)
        count += 1
        print(f"Collected image {count} for class {current_class}")

        # Stop collecting after reaching the number of required images
        if count >= NUM_IMAGES:
            print(f"Finished collecting images for class {current_class}")
            collecting = False

    # Quit the program when 'Esc' is pressed
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        print("Exiting data collection.")
        break

cap.release()
cv2.destroyAllWindows()
