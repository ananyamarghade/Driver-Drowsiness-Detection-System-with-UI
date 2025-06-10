import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Constants for blink detection
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold
CONSEC_FRAMES = 3             # Consecutive frames for a blink
blink_count = 0
frame_count = 0

# Constants for yawn detection
YAWN_THRESH = 0.7             # Yawn threshold
FRAME_COUNT_THRESH = 25        # Yawn frame threshold
yawn_count = 0
yawn_frames = 0
yawning = False

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR)"""
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Eye processing
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])
        
        # Yawn processing
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH])

        # Calculate EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw landmarks
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        # Blink detection logic
        if avg_ear < EAR_THRESHOLD:
            frame_count += 1
        else:
            if frame_count >= CONSEC_FRAMES:
                blink_count += 1
            frame_count = 0

        # Yawn detection logic
        if mar > YAWN_THRESH:
            yawn_frames += 1
            cv2.putText(frame, "Yawning...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if yawn_frames >= FRAME_COUNT_THRESH and not yawning:
                yawn_count += 1
                yawning = True
        else:
            if yawning:
                yawning = False
            yawn_frames = 0

        # Display EAR, MAR, blink, and yawn counts
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Blink and Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
