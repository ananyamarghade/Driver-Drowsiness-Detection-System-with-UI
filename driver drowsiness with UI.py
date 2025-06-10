import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QFrame, QWidget, QLineEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DriverDrowsinessUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #1a1f36; color: white;")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Drowsiness thresholds
        self.EAR_THRESHOLD = 0.21
        self.MAR_THRESHOLD = 0.6
        self.EAR_FRAMES = 15  # threshold to trigger alert
        self.ear_counter = 0

        self.dark_mode = True

        # Data for analysis and graph
        self.ear_values = []
        self.mar_values = []
        self.driver_name = "Driver"

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # UI creation
        self.create_header()
        self.create_content_layout()

        # Start webcam and timer
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_live_feed)
        self.timer.start(30)

    def create_header(self):
        header_layout = QHBoxLayout()
        theme_button = QPushButton()
        theme_button.setIcon(QIcon("moon_icon.png"))
        theme_button.setFixedSize(50, 50)
        theme_button.setStyleSheet("border: none; background-color: transparent;")
        theme_button.clicked.connect(self.toggle_theme)
        header_layout.addWidget(theme_button)

        name_label = QLabel("Enter Driver's Name:")
        name_label.setFont(QFont("Arial", 14))
        name_label.setStyleSheet("color: white;")
        header_layout.addWidget(name_label)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Driver Name")
        self.name_input.setStyleSheet("padding: 5px; font-size: 14px;")
        self.name_input.returnPressed.connect(self.update_driver_name)
        header_layout.addWidget(self.name_input)

        submit_button = QPushButton("Set Name")
        submit_button.setStyleSheet("padding: 5px; font-size: 14px; background-color: green; color: white;")
        submit_button.clicked.connect(self.update_driver_name)
        header_layout.addWidget(submit_button)

        self.main_layout.addLayout(header_layout)

        self.header_label = QLabel(f"Welcome, {self.driver_name}!")
        self.header_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.header_label.setStyleSheet("color: white;")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

    def update_driver_name(self):
        self.driver_name = self.name_input.text() or "Driver"
        self.header_label.setText(f"Welcome, {self.driver_name}!")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.setStyleSheet("background-color: #1a1f36; color: white;")
        else:
            self.setStyleSheet("background-color: white; color: black;")

    def create_content_layout(self):
        content_layout = QHBoxLayout()
        self.main_layout.addLayout(content_layout)

        left_panel = QVBoxLayout()

        self.live_feed_frame = QFrame()
        self.live_feed_frame.setStyleSheet("border: 2px solid #2c2f48;")
        self.live_feed_layout = QVBoxLayout(self.live_feed_frame)

        live_feed_label = QLabel("Live Camera Feed")
        live_feed_label.setFont(QFont("Arial", 14, QFont.Bold))
        live_feed_label.setStyleSheet("color: white;")
        self.live_feed_layout.addWidget(live_feed_label)

        self.live_feed_placeholder = QLabel()
        self.live_feed_placeholder.setStyleSheet("background-color: black;")
        self.live_feed_layout.addWidget(self.live_feed_placeholder)

        self.driver_status = QLabel("NOT DROWSY!")
        self.driver_status.setFont(QFont("Arial", 16, QFont.Bold))
        self.driver_status.setAlignment(Qt.AlignCenter)
        self.driver_status.setStyleSheet("color: green;")
        self.live_feed_layout.addWidget(self.driver_status)

        left_panel.addWidget(self.live_feed_frame)
        content_layout.addLayout(left_panel)

        right_panel = QVBoxLayout()

        analysis_frame = QFrame()
        analysis_frame.setStyleSheet("border: 2px solid #2c2f48;")
        analysis_layout = QVBoxLayout(analysis_frame)

        analysis_label = QLabel("Analysis")
        analysis_label.setFont(QFont("Arial", 14, QFont.Bold))
        analysis_label.setStyleSheet("color: white;")
        analysis_layout.addWidget(analysis_label)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("background-color: #2c2f48; color: white; font-size: 12px;")
        analysis_layout.addWidget(self.analysis_text)

        right_panel.addWidget(analysis_frame)

        graph_frame = QFrame()
        graph_frame.setStyleSheet("border: 2px solid #2c2f48;")
        graph_layout = QVBoxLayout(graph_frame)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("EAR, MAR, and YAW Over Time", color="white")
        self.ax.set_facecolor("#2c2f48")
        self.ax.tick_params(colors="white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["left"].set_color("white")
        graph_layout.addWidget(self.canvas)

        right_panel.addWidget(graph_frame)
        content_layout.addLayout(right_panel)

    def update_live_feed(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (960, 540))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        drowsy_detected = False

        for face in faces:
            shape = self.predictor(gray, face)

            # Landmark visual confirmation (optional)
            for i in range(68):
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 255), -1)

            left_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)])
            right_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)])
            ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
            self.ear_values.append(ear)

            mouth = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            mar = self.mouth_aspect_ratio(mouth)
            self.mar_values.append(mar)

            self.analysis_text.append(f"EAR: {ear:.2f}, MAR: {mar:.2f}")

            cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Threshold logic
            if ear < self.EAR_THRESHOLD:
                self.ear_counter += 1
                if self.ear_counter >= self.EAR_FRAMES:
                    drowsy_detected = True
            else:
                self.ear_counter = 0

            if mar > self.MAR_THRESHOLD:
                drowsy_detected = True

        if drowsy_detected:
            self.driver_status.setText("DROWSY!")
            self.driver_status.setStyleSheet("color: red;")
        else:
            self.driver_status.setText("NOT DROWSY!")
            self.driver_status.setStyleSheet("color: green;")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        q_img = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)
        self.live_feed_placeholder.setPixmap(QPixmap.fromImage(q_img))

        self.update_graph()

    def update_graph(self):
        self.ax.clear()
        self.ax.plot(self.ear_values[-50:], label="EAR", color="yellow")
        self.ax.plot(self.mar_values[-50:], label="MAR", color="cyan")
        self.ax.set_title("EAR and MAR Over Time", color="white")
        self.ax.legend(loc="upper right", facecolor="#2c2f48", edgecolor="white", fontsize=10)
        self.ax.set_facecolor("#2c2f48")
        self.ax.tick_params(colors="white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.canvas.draw()

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    @staticmethod
    def mouth_aspect_ratio(mouth):
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        return (A + B + C) / (3.0 * D)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = DriverDrowsinessUI()
    window.show()
    sys.exit(app.exec_())
