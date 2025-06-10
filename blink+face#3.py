import cv2
import dlib
import numpy as np
import serial
import time
from scipy.spatial import distance as dist

# Arduino communication setup
arduino = serial.Serial('COM3', 9600)  # Change the COM port if necessary
time.sleep(2)  # Allow time for connection

# Constants for blink detection
EAR_THRESHOLD = 0.5  # Eye Aspect Ratio threshold for blink detection
CONSEC_FRAMES = 3      # Number of consecutive frames for blink detection
blink_count = 0
frame_count = 0

# Dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure you have this model

# OpenCV face detection for Arduino communication
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Eye landmark indexes (for dlib's 68 landmarks)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OpenCV face detection for Arduino communication
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Send signals to Arduino
    if len(faces) > 0:
        arduino.write(b'1')  # Send '1' if face detected
    else:
        arduino.write(b'0')  # Send '0' if no face detected

    # Dlib face detection for blink detection
    dlib_faces = detector(gray)

    # Select the largest face (assumed to be the main person)
    largest_face = None
    max_area = 0

    for face in dlib_faces:
        area = (face.right() - face.left()) * (face.bottom() - face.top())
        if area > max_area:
            largest_face = face
            max_area = area

    # Perform blink detection only on the largest face
    if largest_face:
        landmarks = predictor(gray, largest_face)

        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])

        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw face and eye contours
        x, y, w, h = (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green face rectangle

        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)  # Left eye contour
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)  # Right eye contour

        # Blink detection logic
        if avg_ear < EAR_THRESHOLD:
            frame_count += 1
        else:
            if frame_count >= CONSEC_FRAMES:
                blink_count += 1
                frame_count = 0

        # Display EAR and blink count
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw face rectangles from OpenCV detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for Arduino face detection

    # Display the frame
    cv2.imshow("Face & Blink Detection with Arduino", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
