import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Suppress TensorFlow logs (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your pre-trained waste classification model
model = load_model('classifyWaste.h5', compile=False)  # compile=False avoids optimizer/metric warnings

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
# Try different indices if 0 doesn't work (e.g., 1, 2)
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}. Trying next index...")
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}. Please check camera connection.")
        exit()

# Check camera properties for debugging
print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame. Check camera connection.")
        break

    # Debugging: Print frame shape to confirm it's being read
    print(f"Frame {frame_count} shape: {frame.shape}")
    frame_count += 1

    # Preprocess the frame for your model
    # [YOUR INPUT NEEDED]: Replace (224, 224) with the input size your model expects
    input_frame = cv2.resize(frame, (224, 224))
    # Convert to float32 and normalize
    # [YOUR INPUT NEEDED]: Adjust normalization if your model was trained differently
    input_frame = input_frame.astype('float32') / 255.0
    # [YOUR INPUT NEEDED]: Convert to RGB if your model expects RGB instead of BGR
    # input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)  # Uncomment if needed
    # Add batch dimension
    input_frame = np.expand_dims(input_frame, axis=0)

    # Run inference
    predictions = model.predict(input_frame, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Debugging: Print predictions and predicted class
    print(f"Predictions: {predictions[0]}")
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

    # Safeguard: Check if predicted_class is within valid range
    if predicted_class < len(class_labels):
        label = f"{class_labels[predicted_class]} ({confidence:.2f})"
    else:
        label = f"Unknown Class ({confidence:.2f})"

    # Display the prediction on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('Waste Classifier', frame)

    # Ensure the window is visible
    cv2.moveWindow('Waste Classifier', 0, 0)  # Move window to top-left corner

    # Increase delay to ensure the window updates
    key = cv2.waitKey(30) & 0xFF  # Increased from 1ms to 30ms for better rendering
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()