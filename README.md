# Driver-Drowsiness-Detection-System-with-UI

This is a real-time **Driver Drowsiness Detection System** that uses computer vision and facial landmarks to monitor a driver’s eyes. If the driver’s eyes remain closed for a specified time, the system triggers an alert sound, helping to prevent accidents due to fatigue or microsleep.

---

## 📌 About the Project

The goal of this project is to improve road safety by using AI to detect signs of drowsiness in drivers. It continuously analyzes eye aspect ratio (EAR) using a webcam feed. If the system detects prolonged eye closure, it issues an audible warning.

---

## 📁 Project Structure

| File Name                      | Description |
|-------------------------------|-------------|
| `driver drowsiness with UI.py` | Main script with graphical user interface |
| `without UI.py`               | Command-line version of the drowsiness detection system |
| `blink+yawn.py`               | Detects blinking and yawning |
| `head pose.py`                | Monitors head orientation for signs of fatigue |
| `facial landmark.py`          | Visualizes facial landmarks using Dlib |
| `yawn.py`                     | Dedicated yawning detection logic |
| `blink+face#3.py`             | Blinking with face detection logic |
| `requirements.txt`            | Python libraries required for running the project |
| `shape predictor. text`       | A note with the download link for the landmark model (see below) |

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

- `driver drowsiness with UI.py` – the main script for drowsiness detection  
- `shape_predictor_68_face_landmarks.dat` – facial landmark model (⚠️ Not included here due to file size. Download it [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract.)

---

## ▶️ How to Run

1. Download and extract the **shape_predictor_68_face_landmarks.dat** file into your project folder.
2. Install dependencies:
   ```bash
   pip install opencv-python dlib imutils scipy playsound
3. Run the main script:
   driver drowsiness with UI.py
   
---
💡 Final Note
This project represents a small step toward leveraging AI for public safety. Whether you're an enthusiast or a fellow student, feel free to build on this idea and take it further.

Stay safe, stay curious, and keep innovating! 💻🌟
