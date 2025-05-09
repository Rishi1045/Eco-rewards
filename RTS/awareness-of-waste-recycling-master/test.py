import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import time

# Suppress TensorFlow logs (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create a directory to save captured images
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

# Load your pre-trained waste classification model
model = load_model('classifyWaste.h5', compile=False)

# Print model summary and number of output classes for debugging
model.summary()
print(f"Number of output classes in model: {model.output_shape[-1]}")

# Define class labels for your waste categories
class_labels = ["Batteries", "Clothes", "E-waste", "Glass", "Light Bulbs", "Metal", "Organic", "Paper", "Plastic"]
print(f"Number of class labels: {len(class_labels)}")

# Validate that the number of classes matches
if model.output_shape[-1] != len(class_labels):
    print(f"Error: Model output classes ({model.output_shape[-1]}) do not match class_labels ({len(class_labels)}).")
    print("Please update class_labels to match the model's output classes.")
    exit()

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Check camera properties for debugging
print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Create a window with a specific size
cv2.namedWindow('Waste Classifier', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Waste Classifier', 640, 480)

# Variables to store the latest prediction and timing
latest_label = "Waiting 5 seconds to capture..."
capture_count = 0
last_capture_time = time.time()  # Track the last capture time

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Calculate elapsed time since last capture
    current_time = time.time()
    elapsed_time = current_time - last_capture_time

    # Update the label to show countdown
    remaining_time = max(0, 5 - elapsed_time)
    countdown_label = f"Next capture in {remaining_time:.1f}s | {latest_label}"
    cv2.putText(frame, countdown_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Waste Classifier', frame)

    # Ensure the window is visible
    cv2.moveWindow('Waste Classifier', 0, 0)

    # Check for key press to quit
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        break

    # Capture and classify every 5 seconds
    if elapsed_time >= 5:
        capture_count += 1
        last_capture_time = current_time  # Reset the timer

        # Preprocess the captured frame for your model
        input_frame = cv2.resize(frame, (224, 224))  # Replace with your model's input size
        input_frame = input_frame.astype('float32') / 255.0  # Adjust normalization if needed
        # input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)  # Uncomment if model expects RGB
        input_frame = np.expand_dims(input_frame, axis=0)

        # Run inference
        predictions = model.predict(input_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Debugging: Print predictions and predicted class
        print(f"Capture {capture_count} - Predictions: {predictions[0]}")
        print(f"Predicted class: {class_labels[predicted_class]}, Confidence: {confidence:.2f}")

        # Safeguard: Check if predicted_class is within valid range
        if predicted_class < len(class_labels):
            label = f"{class_labels[predicted_class]} ({confidence:.2f})"
        else:
            label = f"Unknown Class ({confidence:.2f})"

        # Update the latest label to display
        latest_label = label

        # Save the captured image with the predicted label
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_images/{class_labels[predicted_class]}_{confidence:.2f}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved captured image as: {filename}")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()