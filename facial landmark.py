import cv2
import dlib
from imutils import face_utils

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for i, face in enumerate(faces):
        # Get the landmarks
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        print(f"\nFace {i+1}: Found {len(shape_np)} landmark points")
        for idx, (x, y) in enumerate(shape_np):
            print(f"Point {idx + 1}: (x={x}, y={y})")

        # Optional: draw landmarks on frame
        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("Facial Landmark Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
