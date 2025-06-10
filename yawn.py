import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Constants
YAWN_THRESH = 0.7  # Yawn detection threshold
FRAME_COUNT_THRESH = 25  # Number of frames yawn should last to be counted

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model

# Yawn counter variables
yawn_count = 0
yawn_frames = 0
yawning = False

def mouth_aspect_ratio(mouth):
    """Calculate the mouth aspect ratio (MAR)"""
    A = dist.euclidean(mouth[2], mouth[10])  # Upper-lower lip distance
    B = dist.euclidean(mouth[4], mouth[8])   # Inner upper-lower lip distance
    C = dist.euclidean(mouth[0], mouth[6])   # Lip width
    mar = (A + B) / (2.0 * C)
    return mar

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract mouth landmarks (points 48-67)
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        
        # Calculate MAR
        mar = mouth_aspect_ratio(mouth)

        # Draw mouth landmarks
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        # Yawn detection logic
        if mar > YAWN_THRESH:
            yawn_frames += 1
            cv2.putText(frame, "Yawning...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Count as a yawn if it lasts for FRAME_COUNT_THRESH frames
            if yawn_frames >= FRAME_COUNT_THRESH and not yawning:
                yawn_count += 1
                yawning = True
        else:
            if yawning:
                yawning = False
            yawn_frames = 0

    # Display yawn count
    cv2.putText(frame, f"Yawn Count: {yawn_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Yawn Detection with Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
