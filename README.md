# Driver-Drowsiness-Detection-System-with-UI

This is a real-time **Driver Drowsiness Detection System** that uses computer vision and facial landmarks to monitor a driver’s eyes. If the driver’s eyes remain closed for a specified time, the system triggers an alert sound, helping to prevent accidents due to fatigue or microsleep.

---

## 📌 About the Project

The goal of this project is to improve road safety by using AI to detect signs of drowsiness in drivers. It continuously analyzes eye aspect ratio (EAR) using a webcam feed. If the system detects prolonged eye closure, it issues an audible warning.

---

## 🧰 Technologies & Tools Used

- **Python**
- **OpenCV** – for capturing and processing video frames
- **Dlib** – for facial landmark detection
- **Imutils** – to simplify image processing tasks
- **Scipy** – for calculating EAR (Eye Aspect Ratio)
- **Playsound** – to trigger alarm when drowsiness is detected
- **shape_predictor_68_face_landmarks.dat** – pre-trained model to extract 68 facial landmarks

---

## 📦 Files in this Project

- `main.py` – the main script for drowsiness detection  
- `alarm.wav` – sound file that plays when drowsiness is detected  
- `shape_predictor_68_face_landmarks.dat` – facial landmark model (⚠️ Not included here due to file size. Download it [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract.)

---

## ▶️ How to Run

1. Download and extract the **shape_predictor_68_face_landmarks.dat** file into your project folder.
2. Install dependencies:
   ```bash
   pip install opencv-python dlib imutils scipy playsound
