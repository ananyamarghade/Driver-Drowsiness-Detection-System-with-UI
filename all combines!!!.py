import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
EAR_FRAMES = 15
YAW_FRAME_THRESHOLD = 25
YAW_DEGREE_THRESHOLD = 20
PITCH_THRESHOLD = 15
YAWN_COUNT_MAR_THRESHOLD = 0.7
YAWN_COUNT_FRAME_THRESHOLD = 25

# Counters
ear_counter = 0
mar_counter = 0
yawn_count = 0
yawn_frames = 0
yawning = False

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Head pose model points
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype="double")

# Camera matrix
focal_length = 1 * 640
center = (640 / 2, 480 / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1))

# EAR function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR function
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[12], mouth[16])
    return (A + B) / (2.0 * C)

# Head pose estimation
def get_head_pose(shape):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye
        (shape.part(45).x, shape.part(45).y),  # Right eye
        (shape.part(48).x, shape.part(48).y),  # Left mouth corner
        (shape.part(54).x, shape.part(54).y)   # Right mouth corner
    ], dtype="double")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # EAR
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # MAR
        mouth = landmarks[48:68]
        mar = mouth_aspect_ratio(mouth)

        # Head pose
        pitch, yaw, roll = get_head_pose(shape)

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # EAR logic
        if ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= EAR_FRAMES:
                cv2.putText(frame, "DROWSY (EAR)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            ear_counter = 0

        # MAR logic for yawning alert
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            if mar_counter >= 10:
                cv2.putText(frame, "Yawning Detected (MAR)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            mar_counter = 0

        # MAR logic for yawn counting
        if mar > YAWN_COUNT_MAR_THRESHOLD:
            yawn_frames += 1
            if yawn_frames >= YAWN_COUNT_FRAME_THRESHOLD and not yawning:
                yawn_count += 1
                yawning = True
        else:
            if yawning:
                yawning = False
            yawn_frames = 0

        # Head pose logic
        if pitch > PITCH_THRESHOLD:
            cv2.putText(frame, "DROWSY (Pitch)", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if abs(yaw) > YAW_DEGREE_THRESHOLD:
            cv2.putText(frame, "ALERT: YAW > 20°", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display data
        cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Yawn Count: {yawn_count}', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Drowsiness and Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
