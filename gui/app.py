import os
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit,
    QProgressBar, QStatusBar, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

STYLESHEET = """
QMainWindow, QWidget { background-color: #0f0f1a; color: #e0e0f0; }
QPushButton { border-radius: 6px; padding: 10px; font-weight: bold; }
QLabel#lbl_video { background-color: #080812; border: 2px solid #1e1e3a; }
"""

class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, str, float)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, predictor, source):
        super().__init__()
        self.predictor = predictor
        self.source = source
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            if isinstance(self.source, int) and sys.platform == "win32":
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.source)
                
            self.predictor.reset_buffer()
            while self._running:
                ret, frame = cap.read()
                if not ret: break
                label, conf, ann_frame, _ = self.predictor.predict_frame(frame)
                self.frame_ready.emit(cv2.resize(ann_frame, (640, 480)), label, conf)
                self.msleep(33)
            cap.release()
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

class HARApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_predictor()
        self._build_ui()
        self.worker = None
        self.source = None

    def _init_predictor(self):
        from utils.predictor import ActivityPredictor
        self.predictor = ActivityPredictor()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.lbl_video = QLabel()
        self.lbl_video.setMinimumSize(640, 480)
        layout.addWidget(self.lbl_video)

        btn_box = QHBoxLayout()
        btn_upload = QPushButton("Upload Video")
        btn_upload.clicked.connect(self._upload)
        btn_box.addWidget(btn_upload)
        
        btn_webcam = QPushButton("Start Webcam")
        btn_webcam.clicked.connect(self._start_webcam)
        btn_box.addWidget(btn_webcam)
        
        layout.addLayout(btn_box)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        layout.addWidget(self.btn_stop)
        
        self.lbl_activity = QLabel("Ready")
        layout.addWidget(self.lbl_activity)

    def _upload(self):
        file, _ = QFileDialog.getOpenFileName()
        if file:
            self.source = file
            self._start()

    def _start_webcam(self):
        self.source = 0
        self._start()
        
    def _start(self):
        self._stop()
        self.worker = VideoWorker(self.predictor, self.source)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def _on_frame(self, frame, label, conf):
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, w*3, QImage.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(q_img))
        self.lbl_activity.setText(f"{label} ({conf:.2f})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HARApplication()
    w.show()
    sys.exit(app.exec_())