import cv2
import os
import numpy as np
from keras.models import load_model
import dlib

# Replace with the correct absolute path to the model file
model_path = "C:\\Users\\nagar\\Downloads\\Project\\model_v6_23.hdf5"

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at path: {model_path}")
    exit()

# Load the pre-trained emotion recognition model
emotion_classifier = load_model(model_path, compile=False)

# Check if the model loaded successfully
if emotion_classifier is None:
    print("Error: Failed to load the emotion recognition model.")
    exit()

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
existing_predictor_path = "C:\\Users\\nagar\\Downloads\\new_shape_predictor.dat"

# Load the predictor using dlib.shape_predictor directly
predictor = dlib.shape_predictor(existing_predictor_path)

# Define the list of emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to detect facial expression
def detect_expression(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Handle case when no faces are detected
    if len(faces) == 0:
        print("No face detected in the frame.")
        return frame

    for face in faces:
        landmarks = predictor(gray, face)
        shape = np.array([[point.x, point.y] for point in landmarks.parts()])  # Convert landmarks to NumPy array

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        face_roi = gray[y:y + h, x:x + w]

        # Check if face_roi is empty before processing
        if face_roi.size == 0:
            print("Empty face_roi, skipping this face.")
            continue

        # Apply histogram equalization for better results
        face_roi = cv2.equalizeHist(face_roi)

        # Resize the face ROI to match the input size for emotion detection model
        roi = cv2.resize(face_roi, (48, 48))  # Resize to (48, 48)
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
        roi = np.expand_dims(roi, axis=0) / 255.0  # Add batch dimension and normalize

        # Predict the emotion
        emotion_preds = emotion_classifier.predict(roi)[0]
        emotion_label = emotions[np.argmax(emotion_preds)]

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Check if the video capture was successful
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from webcam.")
        break

    # Detect facial expressions in the frame
    frame = detect_expression(frame)

    # Display the frame
    cv2.imshow('Facial Expression Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
print("Video capture released and windows closed.")
