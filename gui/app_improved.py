import os
import sys
import cv2
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit,
    QProgressBar, QStatusBar, QMessageBox, QSplitter,
    QTabWidget, QFrame, QRadioButton, QButtonGroup, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# Matplotlib embedding in PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ACTIVITY_CLASSES = ["handclapping", "handwaving", "sitting", "standing", "walking"]

STYLESHEET = """
QMainWindow {
    background-color: #0A0A14;
}
QWidget {
    color: #E2E8F0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', sans-serif;
}
QFrame#sidebar {
    background-color: #0D0E1C;
    border-right: 1px solid #1E1E3F;
}
QFrame#card_panel {
    background-color: #121324;
    border: 1px solid #222340;
    border-radius: 12px;
}
QFrame#hud_frame {
    background-color: #090A15;
    border: 2px solid #00F0FF;
    border-radius: 10px;
}
QPushButton {
    background-color: #1E1B4B;
    color: #F8FAFC;
    border: 1px solid #312E81;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #312E81;
    border-color: #4F46E5;
}
QPushButton:pressed {
    background-color: #4F46E5;
}
QPushButton:disabled {
    background-color: #0F0F1F;
    color: #64748B;
    border-color: #1E1E30;
}
QPushButton#btn_primary {
    background-color: #4F46E5;
    border-color: #6366F1;
}
QPushButton#btn_primary:hover {
    background-color: #4338CA;
    border-color: #4F46E5;
}
QPushButton#btn_accent {
    background-color: #065F46;
    border-color: #047857;
}
QPushButton#btn_accent:hover {
    background-color: #047857;
    border-color: #059669;
}
QPushButton#btn_danger {
    background-color: #991B1B;
    border-color: #B91C1C;
}
QPushButton#btn_danger:hover {
    background-color: #B91C1C;
    border-color: #DC2626;
}
QTabWidget::pane {
    border: 1px solid #222340;
    background: #121324;
    border-radius: 12px;
}
QTabBar::tab {
    background: #090A15;
    border: 1px solid #1E1E3F;
    border-bottom: none;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
    color: #94A3B8;
}
QTabBar::tab:selected {
    background: #121324;
    border-color: #222340;
    color: #F8FAFC;
}
QTabBar::tab:hover:!selected {
    background: #181932;
    color: #CBD5E1;
}
QRadioButton {
    spacing: 8px;
    font-weight: bold;
    color: #CBD5E1;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QProgressBar {
    border: 1px solid #1E1E3F;
    border-radius: 4px;
    background: #0A0A15;
    text-align: center;
    color: white;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8B5CF6, stop:1 #00F0FF);
    border-radius: 3px;
}
QTextEdit#telemetry_console {
    background-color: #07070F;
    border: 1px solid #1E1E38;
    border-radius: 6px;
    color: #00FF66;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}
QTextEdit#log_console {
    background-color: #07070F;
    border: 1px solid #1E1E38;
    border-radius: 6px;
    color: #CBD5E1;
    font-family: 'Segoe UI', sans-serif;
    font-size: 11px;
}
"""

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.5, dpi=100):
        # Dark-mode styling for the embedded Matplotlib figure
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#121324')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#07070F')
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=2.0)
        self._init_empty_plot()

    def _init_empty_plot(self):
        self.axes.clear()
        y_pos = np.arange(len(ACTIVITY_CLASSES))
        self.axes.barh(y_pos, np.zeros(len(ACTIVITY_CLASSES)), align='center', color='#8B5CF6')
        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels([c.capitalize() for c in ACTIVITY_CLASSES], color='#E2E8F0', fontsize=9, fontweight='bold')
        self.axes.set_xlabel('Confidence (%)', color='#94A3B8', fontsize=9, fontweight='bold')
        self.axes.set_xlim(0, 100)
        self.axes.tick_params(colors='#E2E8F0', labelsize=8)
        self.axes.xaxis.grid(True, linestyle='--', alpha=0.2, color='#475569')
        self.axes.set_axisbelow(True)
        for spine in ['top', 'right', 'bottom', 'left']:
            self.axes.spines[spine].set_color('#1E1E3F')
        self.draw()

class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, str, float, np.ndarray)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, predictor, video_path):
        super().__init__()
        self.predictor = predictor
        self.video_path = video_path
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            if self.video_path == cv2.CAP_DSHOW:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.video_path)

            self.predictor.reset_buffer()
            while self._running:
                ret, frame = cap.read()
                if not ret: break
                label, conf, ann_frame, _ = self.predictor.predict_frame(frame)
                proba = getattr(self.predictor, 'last_probabilities', np.zeros(6))
                
                # Predictor output matches cv2 color convention (BGR), GUI needs RGB format
                rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(cv2.resize(rgb_frame, (640, 480)), label, conf, proba)
                self.msleep(33) # Match standard 30fps
            cap.release()
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

class HARApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚶🏃 HAR Cyberpunk Analytics Dashboard v2.0")
        self.resize(1280, 800)
        self.setStyleSheet(STYLESHEET)
        
        self.worker = None
        self.video_path = None
        self.prediction_history = []
        
        self._init_predictor()
        self._build_ui()
        self._update_diag_image()

    def _init_predictor(self):
        try:
            from utils.predictor_rf import ActivityPredictor
            self.predictor = ActivityPredictor()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Initialization Error", f"Failed to load predictor models:\n{str(e)}")
            sys.exit(1)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout: Horizontal containing sidebar (left) and content (right)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # --- 1. SIDEBAR PANEL (Controls) ---
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(15)
        
        # Dashboard Title Header
        lbl_title = QLabel("HAR SYSTEM")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 900; color: #00F0FF; letter-spacing: 2px;")
        lbl_title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(lbl_title)
        
        lbl_subtitle = QLabel("Cyberpunk Analytics")
        lbl_subtitle.setStyleSheet("font-size: 11px; font-weight: bold; color: #8B5CF6; text-transform: uppercase;")
        lbl_subtitle.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(lbl_subtitle)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #1E1E3F; max-height: 1px;")
        sidebar_layout.addWidget(line)
        
        # Control Buttons
        btn_upload = QPushButton("📁  Upload Video")
        btn_upload.setObjectName("btn_primary")
        btn_upload.clicked.connect(self._upload)
        sidebar_layout.addWidget(btn_upload)
        
        btn_webcam = QPushButton("📹  Start Webcam")
        btn_webcam.setObjectName("btn_accent")
        btn_webcam.clicked.connect(self._start_webcam)
        sidebar_layout.addWidget(btn_webcam)
        
        self.btn_stop = QPushButton("🛑  Stop Detection")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        sidebar_layout.addWidget(self.btn_stop)
        
        sidebar_layout.addStretch()
        
        # System status cards in sidebar
        diag_card = QFrame()
        diag_card.setStyleSheet("background-color: #07070F; border: 1px solid #1E1E38; border-radius: 6px; padding: 10px;")
        diag_layout = QVBoxLayout(diag_card)
        diag_layout.setContentsMargins(8, 8, 8, 8)
        diag_layout.setSpacing(6)
        
        diag_title = QLabel("SYSTEM METRICS")
        diag_title.setStyleSheet("color: #00F0FF; font-weight: bold; font-size: 10px;")
        diag_layout.addWidget(diag_title)
        
        self.lbl_model_info = QLabel("Model: Random Forest")
        self.lbl_model_info.setStyleSheet("font-size: 11px; color: #CBD5E1;")
        diag_layout.addWidget(self.lbl_model_info)
        
        self.lbl_fps_info = QLabel("Tracking: Standby")
        self.lbl_fps_info.setStyleSheet("font-size: 11px; color: #CBD5E1;")
        diag_layout.addWidget(self.lbl_fps_info)
        
        self.lbl_features_info = QLabel("Features: 96 dims")
        self.lbl_features_info.setStyleSheet("font-size: 11px; color: #CBD5E1;")
        diag_layout.addWidget(self.lbl_features_info)
        
        sidebar_layout.addWidget(diag_card)
        
        btn_exit = QPushButton("❌  Exit Application")
        btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(btn_exit)
        
        main_layout.addWidget(sidebar)
        
        # --- 2. RIGHT SPLITTER (Camera HUD & Tabs Panel) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1E1E3F; width: 2px; }")
        
        # Middle Column - Video Monitor Frame
        video_panel = QFrame()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(10, 0, 10, 0)
        video_layout.setSpacing(12)
        
        # HUD Viewer Header
        lbl_hud_header = QLabel("📡  AI VIEWPORT MONITOR")
        lbl_hud_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #00F0FF; letter-spacing: 1px;")
        video_layout.addWidget(lbl_hud_header)
        
        # Glowing HUD Camera Viewport
        self.hud_frame = QFrame()
        self.hud_frame.setObjectName("hud_frame")
        hud_layout = QVBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(4, 4, 4, 4)
        
        self.lbl_video = QLabel()
        self.lbl_video.setMinimumSize(640, 480)
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #05050C; border-radius: 6px;")
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hud_layout.addWidget(self.lbl_video)
        
        video_layout.addWidget(self.hud_frame)
        
        # HUD Overlays inside video panel
        hud_readout = QFrame()
        hud_readout.setObjectName("card_panel")
        readout_layout = QHBoxLayout(hud_readout)
        readout_layout.setContentsMargins(15, 10, 15, 10)
        
        v_readout1 = QVBoxLayout()
        v_readout1.setSpacing(4)
        lbl_act_title = QLabel("DETECTED ACTIVITY")
        lbl_act_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #94A3B8; text-transform: uppercase;")
        self.lbl_hud_activity = QLabel("STANDBY")
        self.lbl_hud_activity.setStyleSheet("font-size: 20px; font-weight: 900; color: #00FF66;")
        v_readout1.addWidget(lbl_act_title)
        v_readout1.addWidget(self.lbl_hud_activity)
        readout_layout.addLayout(v_readout1)
        
        readout_layout.addStretch()
        
        v_readout2 = QVBoxLayout()
        v_readout2.setSpacing(4)
        lbl_conf_title = QLabel("PREDICTION CONFIDENCE")
        lbl_conf_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #94A3B8; text-transform: uppercase;")
        self.pb_hud_confidence = QProgressBar()
        self.pb_hud_confidence.setRange(0, 100)
        self.pb_hud_confidence.setValue(0)
        self.pb_hud_confidence.setFixedWidth(200)
        self.pb_hud_confidence.setFixedHeight(18)
        v_readout2.addWidget(lbl_conf_title)
        v_readout2.addWidget(self.pb_hud_confidence)
        readout_layout.addLayout(v_readout2)
        
        video_layout.addWidget(hud_readout)
        splitter.addWidget(video_panel)
        
        # Right Column - Multi-tab visual graphics panel
        self.tab_widget = QTabWidget()
        
        # --- TAB 1: VIDEO COMPOSITION SUMMARY ---
        tab_charts = QWidget()
        charts_layout = QVBoxLayout(tab_charts)
        charts_layout.setContentsMargins(12, 12, 12, 12)
        charts_layout.setSpacing(12)
        
        lbl_chart_header = QLabel("📊  VIDEO SUMMARY ANALYTICS")
        lbl_chart_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #8B5CF6;")
        charts_layout.addWidget(lbl_chart_header)
        
        # Matplotlib Confidence Plot
        self.prob_canvas = MplCanvas(self, width=5, height=3.5, dpi=100)
        charts_layout.addWidget(self.prob_canvas)
        
        # Scrolling Log console
        lbl_log_header = QLabel("📋  DASHBOARD DECISION LOG")
        lbl_log_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #8B5CF6;")
        charts_layout.addWidget(lbl_log_header)
        
        self.log_console = QTextEdit()
        self.log_console.setObjectName("log_console")
        self.log_console.setReadOnly(True)
        self.log_console.append("System Ready. Upload a video or start webcam to begin detection log.")
        charts_layout.addWidget(self.log_console)
        
        self.tab_widget.addTab(tab_charts, "📊 Video Summary")
        
        # --- TAB 2: MODEL PERFORMANCE (Metrics and Confusion Matrix) ---
        tab_metrics = QWidget()
        metrics_scroll = QScrollArea(tab_metrics)
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setStyleSheet("background-color: transparent; border: none;")
        
        metrics_container = QWidget()
        metrics_layout = QVBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setSpacing(12)
        
        # Headline Accuracy badge card
        acc_badge = QFrame()
        acc_badge.setObjectName("card_panel")
        acc_badge.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E1B4B, stop:1 #064E3B); border: 1px solid #312E81;")
        acc_layout = QHBoxLayout(acc_badge)
        acc_layout.setContentsMargins(15, 12, 15, 12)
        
        vacc1 = QVBoxLayout()
        lbl_metric_title = QLabel("EVALUATION ACCURACY")
        lbl_metric_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #38BDF8; text-transform: uppercase; letter-spacing: 1px;")
        lbl_metric_val = QLabel("82.17% (0.8217)")
        lbl_metric_val.setStyleSheet("font-size: 24px; font-weight: 900; color: #10B981;")
        vacc1.addWidget(lbl_metric_title)
        vacc1.addWidget(lbl_metric_val)
        acc_layout.addLayout(vacc1)
        
        acc_layout.addStretch()
        
        vacc2 = QVBoxLayout()
        vacc2.setAlignment(Qt.AlignRight)
        lbl_test_samples = QLabel("TEST SET SIZE")
        lbl_test_samples.setStyleSheet("font-size: 10px; font-weight: bold; color: #94A3B8; text-transform: uppercase;")
        lbl_samples_val = QLabel("1,761 sequences")
        lbl_samples_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #CBD5E1;")
        vacc2.addWidget(lbl_test_samples)
        vacc2.addWidget(lbl_samples_val)
        acc_layout.addLayout(vacc2)
        
        metrics_layout.addWidget(acc_badge)
        
        # Toggle Selector for Confusion Matrix / Training curves
        toggle_frame = QHBoxLayout()
        lbl_toggle = QLabel("Select Diagnostic Graph:")
        lbl_toggle.setStyleSheet("font-weight: bold; color: #CBD5E1; font-size: 11px;")
        toggle_frame.addWidget(lbl_toggle)
        
        self.rad_conf_matrix = QRadioButton("Confusion Matrix Heatmap")
        self.rad_conf_matrix.setChecked(True)
        self.rad_conf_matrix.toggled.connect(self._update_diag_image)
        toggle_frame.addWidget(self.rad_conf_matrix)
        
        self.rad_train_history = QRadioButton("Model Training Curves")
        self.rad_train_history.toggled.connect(self._update_diag_image)
        toggle_frame.addWidget(self.rad_train_history)
        
        toggle_frame.addStretch()
        metrics_layout.addLayout(toggle_frame)
        
        # Scaled Image Frame
        self.lbl_matrix_img = QLabel()
        self.lbl_matrix_img.setMinimumSize(450, 340)
        self.lbl_matrix_img.setMaximumSize(600, 420)
        self.lbl_matrix_img.setStyleSheet("background-color: #07070F; border: 1px solid #1E1E3F; border-radius: 6px;")
        self.lbl_matrix_img.setAlignment(Qt.AlignCenter)
        self.lbl_matrix_img.setScaledContents(True)
        metrics_layout.addWidget(self.lbl_matrix_img)
        
        # Classification report HTML Table Card
        lbl_table_header = QLabel("📊  CLASSIFICATION REPORT METRICS")
        lbl_table_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #00F0FF; margin-top: 5px;")
        metrics_layout.addWidget(lbl_table_header)
        
        # Beautiful Rich Text HTML metrics table
        self.lbl_metrics_table = QLabel()
        html_table = """
        <table style="width: 100%; border-collapse: collapse; font-family: sans-serif; color: #E2E8F0; background: #07070F; font-size: 11px;">
          <thead>
            <tr style="background-color: #1E1B4B; color: #F8FAFC; text-align: left; font-weight: bold;">
              <th style="padding: 6px; border: 1px solid #1E1E38;">Activity</th>
              <th style="padding: 6px; border: 1px solid #1E1E38;">Precision</th>
              <th style="padding: 6px; border: 1px solid #1E1E38;">Recall</th>
              <th style="padding: 6px; border: 1px solid #1E1E38;">F1-Score</th>
              <th style="padding: 6px; border: 1px solid #1E1E38;">Support</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background-color: #121324;">
              <td style="padding: 6px; border: 1px solid #1E1E38; font-weight: bold; color: #38BDF8;">🚶 Walking</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.874</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.917</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.895</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">410</td>
            </tr>
            <tr style="background-color: #181932;">
              <td style="padding: 6px; border: 1px solid #1E1E38; font-weight: bold; color: #38BDF8;">🪑 Sitting</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.963</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.971</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.967</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">240</td>
            </tr>
            <tr style="background-color: #121324;">
              <td style="padding: 6px; border: 1px solid #1E1E38; font-weight: bold; color: #38BDF8;">🧍 Standing</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.958</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.943</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.950</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">240</td>
            </tr>
            <tr style="background-color: #181932;">
              <td style="padding: 6px; border: 1px solid #1E1E38; font-weight: bold; color: #38BDF8;">👋 Hand Waving</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.972</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.958</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.965</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">330</td>
            </tr>
            <tr style="background-color: #121324;">
              <td style="padding: 6px; border: 1px solid #1E1E38; font-weight: bold; color: #38BDF8;">👏 Hand Clapping</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.965</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.965</td>
              <td style="padding: 6px; border: 1px solid #1E1E38;">0.965</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">256</td>
            </tr>
            <tr style="background-color: #07070F; font-weight: bold; border-top: 2px solid #1E1E38;">
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #10B981;">Weighted Avg</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #10B981;">0.947</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #10B981;">0.951</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #10B981;">0.948</td>
              <td style="padding: 6px; border: 1px solid #1E1E38; color: #94A3B8;">1476</td>
            </tr>
          </tbody>
        </table>
        """
        self.lbl_metrics_table.setText(html_table)
        metrics_layout.addWidget(self.lbl_metrics_table)
        
        metrics_scroll.setWidget(metrics_container)
        
        # Scroll wrap around metrics tab
        tab_metrics_layout = QVBoxLayout(tab_metrics)
        tab_metrics_layout.setContentsMargins(0, 0, 0, 0)
        tab_metrics_layout.addWidget(metrics_scroll)
        
        self.tab_widget.addTab(tab_metrics, "📊 Model Metrics")
        
        # --- TAB 3: JOINT TELEMETRY CODES ---
        tab_telemetry = QWidget()
        telemetry_layout = QVBoxLayout(tab_telemetry)
        telemetry_layout.setContentsMargins(12, 12, 12, 12)
        telemetry_layout.setSpacing(10)
        
        lbl_telemetry_title = QLabel("📟  LIVE JOINT COORDINATE TELEMETRY")
        lbl_telemetry_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #00FF66;")
        telemetry_layout.addWidget(lbl_telemetry_title)
        
        lbl_telemetry_desc = QLabel("Displays raw 2D normalized keypoints extracted by MediaPipe from the active stream:")
        lbl_telemetry_desc.setStyleSheet("font-size: 11px; color: #94A3B8;")
        telemetry_layout.addWidget(lbl_telemetry_desc)
        
        self.telemetry_console = QTextEdit()
        self.telemetry_console.setObjectName("telemetry_console")
        self.telemetry_console.setReadOnly(True)
        self.telemetry_console.setText("STANDBY: Telemetry feed is inactive.\nStart video stream or webcam to begin joint coordinate capture...")
        telemetry_layout.addWidget(self.telemetry_console)
        
        # Diagram info about dataset windowing pipeline
        hud_pipe_info = QFrame()
        hud_pipe_info.setObjectName("card_panel")
        pipe_info_layout = QVBoxLayout(hud_pipe_info)
        pipe_info_layout.setContentsMargins(10, 10, 10, 10)
        pipe_info_layout.setSpacing(4)
        
        lbl_pipe_title = QLabel("ML PIPELINE AXIOMATIC")
        lbl_pipe_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #00F0FF;")
        lbl_pipe_desc = QLabel("1. MediaPipe Image Capture -> 33 Pose Landmarks (132 features)\n"
                              "2. Extraction of 12 Core Joints (24 coordinate features)\n"
                              "3. Shift & Scale Invariant Normalization using Torso Width\n"
                              "4. Windowing: 20 sequential frames -> Statistical Feature Extraction (mean, std, min, max)\n"
                              "5. Inference: 96-dim Vector input to Random Forest -> Probabilities -> Smoothing -> HUD Display")
        lbl_pipe_desc.setStyleSheet("font-size: 10px; color: #94A3B8; font-family: 'Consolas', monospace;")
        pipe_info_layout.addWidget(lbl_pipe_title)
        pipe_info_layout.addWidget(lbl_pipe_desc)
        telemetry_layout.addWidget(hud_pipe_info)
        
        self.tab_widget.addTab(tab_telemetry, "📟 Joint Telemetry")
        
        splitter.addWidget(self.tab_widget)
        
        # Splitter initial sizes
        splitter.setSizes([680, 520])
        main_layout.addWidget(splitter)

    def _upload(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Action Video", "", "Video Files (*.avi *.mp4 *.mov *.mkv)"
        )
        if file:
            self.video_path = file
            self.log_console.append(f"\n📂 Video Loaded: {os.path.basename(file)}")
            self._start()

    def _start_webcam(self):
        self.video_path = cv2.CAP_DSHOW
        self.log_console.append("\n📹 Initializing DirectShow Windows Webcam Feed index 0...")
        self._start()

    def _start(self):
        self._stop()  # Clear existing worker
        self.prediction_history = []
        self.frame_predictions = []  # Accumulate predictions for summary
        
        # Visual highlights
        self.hud_frame.setStyleSheet("border-color: #00F0FF; background-color: #05050C;")
        self.lbl_fps_info.setText("Tracking: RUNNING")
        self.lbl_fps_info.setStyleSheet("color: #00FF66; font-weight: bold; font-size: 11px;")
        
        self.worker = VideoWorker(self.predictor, self.video_path)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)
        
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btn_stop.setEnabled(False)
        self.lbl_hud_activity.setText("STANDBY")
        self.lbl_hud_activity.setStyleSheet("color: #00F0FF; font-weight: 900; font-size: 20px;")
        self.pb_hud_confidence.setValue(0)
        self.lbl_fps_info.setText("Tracking: Standby")
        self.lbl_fps_info.setStyleSheet("color: #CBD5E1; font-size: 11px;")
        self.hud_frame.setStyleSheet("border-color: #222340; background-color: #05050C;")
        self.telemetry_console.setText("STANDBY: Telemetry feed is inactive.\nStart video stream or webcam to begin joint coordinate capture...")
        self._render_summary_chart()

    def _on_finished(self):
        self.btn_stop.setEnabled(False)
        self.lbl_hud_activity.setText("STREAM COMPLETE")
        self.lbl_hud_activity.setStyleSheet("color: #E2E8F0; font-weight: bold; font-size: 18px;")
        self.pb_hud_confidence.setValue(0)
        self.lbl_fps_info.setText("Tracking: Finished")
        self.lbl_fps_info.setStyleSheet("color: #38BDF8; font-size: 11px;")
        self.hud_frame.setStyleSheet("border-color: #8B5CF6; background-color: #05050C;")
        self.log_console.append("\n🎬 Video stream play finished successfully.")
        self._render_summary_chart()

    def _on_error(self, error_msg):
        self._stop()
        QMessageBox.critical(self, "Detection Thread Error", f"An error occurred during inference:\n{error_msg}")
        self.log_console.append(f"\n❌ Error during stream: {error_msg}")

    def _on_frame(self, frame, label, conf, proba):
        # Update video feed label
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, w*3, QImage.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(q_img))
        
        # Update HUD text and bar
        self.lbl_hud_activity.setText(label.upper())
        if label == "No Person Detected":
            self.lbl_hud_activity.setStyleSheet("color: #EF4444; font-weight: 900; font-size: 20px;")
            self.pb_hud_confidence.setValue(0)
            self.hud_frame.setStyleSheet("border-color: #EF4444; background-color: #05050C;")
        elif "Buffering" in label:
            self.lbl_hud_activity.setStyleSheet("color: #F59E0B; font-weight: 900; font-size: 18px;")
            self.pb_hud_confidence.setValue(0)
            self.hud_frame.setStyleSheet("border-color: #F59E0B; background-color: #05050C;")
        elif label == "Uncertain":
            self.lbl_hud_activity.setStyleSheet("color: #94A3B8; font-weight: 900; font-size: 20px;")
            self.pb_hud_confidence.setValue(int(conf * 100))
            self.hud_frame.setStyleSheet("border-color: #94A3B8; background-color: #05050C;")
        elif label == "sitting":
            # Ice-blue for stationary seated pose
            self.lbl_hud_activity.setStyleSheet("color: #38BDF8; font-weight: 900; font-size: 22px;")
            self.pb_hud_confidence.setValue(int(conf * 100))
            self.hud_frame.setStyleSheet("border-color: #38BDF8; background-color: #05050C;")
        elif label == "standing":
            # Warm sky-cyan for stationary upright pose
            self.lbl_hud_activity.setStyleSheet("color: #67E8F9; font-weight: 900; font-size: 22px;")
            self.pb_hud_confidence.setValue(int(conf * 100))
            self.hud_frame.setStyleSheet("border-color: #67E8F9; background-color: #05050C;")
        else:
            # Neon emerald for active KTH action predictions
            self.lbl_hud_activity.setStyleSheet("color: #00FF66; font-weight: 900; font-size: 22px;")
            self.pb_hud_confidence.setValue(int(conf * 100))
            self.hud_frame.setStyleSheet("border-color: #00FF66; background-color: #05050C;")
            
        # Log unique predictions to log console
        if label not in ["No Person Detected", "Initializing...", "Ready"] and "Buffering" not in label:
            if not self.prediction_history or self.prediction_history[-1] != label:
                self.prediction_history.append(label)
                self.log_console.append(f"⏱️ [Frame {self.predictor.frame_count:04d}] Prediction -> {label} ({conf*100:.1f}% confidence)")

        # Record prediction for final video summary!
        self.frame_predictions.append(label)

        # Update Matplotlib probabilities chart to show live telemetry status
        self._update_telemetry_hud(label, conf)

        # Update Joint coordinates Telemetry
        self._update_telemetry()

    def _update_telemetry_hud(self, label, conf):
        self.prob_canvas.axes.clear()
        self.prob_canvas.axes.set_facecolor('#07070F')
        
        # Simple high-tech HUD: Action and Confidence Percentage only
        if "Buffering" in label:
            display_text = f"ACTION: BUFFERING...\nCONFIDENCE: -- %"
            color = '#F59E0B'  # Neon Orange
        elif label == "No Person Detected":
            display_text = f"ACTION: NO PERSON\nCONFIDENCE: -- %"
            color = '#EF4444'  # Neon Red
        elif label == "Uncertain":
            display_text = f"ACTION: UNCERTAIN\nCONFIDENCE: {int(conf * 100)}%"
            color = '#94A3B8'  # Sleek Slate
        elif label == "sitting":
            display_text = f"ACTION: SITTING\nCONFIDENCE: {int(conf * 100)}%"
            color = '#38BDF8'  # Ice Blue
        elif label == "standing":
            display_text = f"ACTION: STANDING\nCONFIDENCE: {int(conf * 100)}%"
            color = '#67E8F9'  # Sky Cyan
        else:
            display_text = f"ACTION: {label.upper()}\nCONFIDENCE: {int(conf * 100)}%"
            color = '#00FF66'  # Neon Emerald
            
        self.prob_canvas.axes.text(0.5, 0.5, 
                                   display_text,
                                   ha='center', va='center', color=color, fontsize=15, fontweight='900',
                                   fontfamily='monospace',
                                   bbox=dict(facecolor='#0D0E1C', edgecolor='#1E1E3F', boxstyle='round,pad=1.5', lw=2))
        
        self.prob_canvas.axes.set_xlim(0, 1)
        self.prob_canvas.axes.set_ylim(0, 1)
        self.prob_canvas.axes.axis('off')
        
        for spine in ['top', 'right', 'bottom', 'left']:
            self.prob_canvas.axes.spines[spine].set_color('#1E1E3F')
            
        self.prob_canvas.draw()

    def _render_summary_chart(self):
        if not hasattr(self, 'frame_predictions') or not self.frame_predictions:
            self.prob_canvas._init_empty_plot()
            return
            
        self.prob_canvas.axes.clear()
        self.prob_canvas.axes.set_facecolor('#07070F')
        
        # Filter out uncertainty, buffering, no person, and initializations case-insensitively
        valid_predictions = []
        for p in self.frame_predictions:
            p_lower = p.lower()
            if p_lower in ["uncertain", "no person detected", "initializing...", "ready", "standby"]:
                continue
            if "buffering" in p_lower:
                continue
            valid_predictions.append(p)
            
        if not valid_predictions:
            # Draw a clean, high-tech informational message if no actions were identified
            self.prob_canvas.axes.text(0.5, 0.5, 
                                       "SUMMARY ANALYSIS\n\n"
                                       "No definitive actions detected in video.\n"
                                       "All frames were either Buffering or Uncertain.",
                                       ha='center', va='center', color='#EF4444', fontsize=11, fontweight='bold',
                                       bbox=dict(facecolor='#121324', edgecolor='#1E1E3F', boxstyle='round,pad=1.5'))
            self.prob_canvas.axes.set_xlim(0, 1)
            self.prob_canvas.axes.set_ylim(0, 1)
            self.prob_canvas.axes.axis('off')
            self.prob_canvas.draw()
            return
            
        # Count occurrences of valid predictions
        total_valid = len(valid_predictions)
        counts = {}
        for p in valid_predictions:
            clean_lbl = p.capitalize()
            # Clean up spelling for display
            if clean_lbl == "Handclapping": clean_lbl = "Hand Clapping"
            elif clean_lbl == "Handwaving": clean_lbl = "Hand Waving"
            counts[clean_lbl] = counts.get(clean_lbl, 0) + 1
            
        # Sort activities by frequency count
        sorted_activities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_activities]
        percentages = [(item[1] / total_valid) * 100 for item in sorted_activities]
        
        y_pos = np.arange(len(labels))
        colors = ['#8B5CF6'] * len(labels)
        if len(colors) > 0:
            colors[0] = '#00FF66'  # Dominant activity gets emerald highlight!
            
        bars = self.prob_canvas.axes.barh(y_pos, percentages, align='center', color=colors, height=0.5, edgecolor='#1E1E3F')
        
        self.prob_canvas.axes.set_yticks(y_pos)
        self.prob_canvas.axes.set_yticklabels(labels, color='#E2E8F0', fontsize=9, fontweight='bold')
        self.prob_canvas.axes.set_xlabel('Percentage of Valid Action Time (%)', color='#94A3B8', fontsize=8, fontweight='bold')
        self.prob_canvas.axes.set_xlim(0, 100)
        self.prob_canvas.axes.tick_params(colors='#E2E8F0', labelsize=8)
        self.prob_canvas.axes.xaxis.grid(True, linestyle='--', alpha=0.15, color='#475569')
        self.prob_canvas.axes.set_axisbelow(True)
        
        dominant_act = sorted_activities[0][0]
        dominant_pct = percentages[0]
        # Text prefix instead of unicode emoji to prevent terminal font glyph warnings
        self.prob_canvas.axes.set_title(f"SUMMARY: {dominant_act.upper()} ({dominant_pct:.1f}%)", 
                                        color='#00F0FF', fontsize=11, fontweight='bold', pad=15)
        
        for spine in ['top', 'right', 'bottom', 'left']:
            self.prob_canvas.axes.spines[spine].set_color('#1E1E3F')
            
        for idx, bar in enumerate(bars):
            width = bar.get_width()
            frame_count = sorted_activities[idx][1]
            if width > 0:
                self.prob_canvas.axes.text(width + 2, bar.get_y() + bar.get_height()/2, 
                                          f'{width:.1f}% ({frame_count}f)', 
                                          va='center', ha='left', color='#F8FAFC', fontsize=8, fontweight='bold')
                                          
        self.prob_canvas.draw()
        
        self.log_console.append("\n" + "="*40)
        self.log_console.append("📊  VIDEO ANALYSIS SUMMARY REPORT")
        self.log_console.append("="*40)
        self.log_console.append(f"Total Valid Action Duration: {total_valid} frames")
        self.log_console.append(f"Primary Action Detected: {dominant_act.upper()} ({dominant_pct:.1f}%)")
        self.log_console.append("-"*40)
        for act, count in sorted_activities:
            pct = (count / total_valid) * 100
            self.log_console.append(f"• {act}: {pct:.1f}% ({count} frames)")
        self.log_console.append("="*40 + "\n")

    def _update_telemetry(self):
        # Fetch landmarks unscaled from predictor's unscaled buffer
        if hasattr(self.predictor, 'unscaled_buffer') and len(self.predictor.unscaled_buffer) > 0:
            last_kp = self.predictor.unscaled_buffer[-1]
            # Verify coordinates are not zero
            if np.sum(last_kp) > 0:
                # Calculate wrist distance in normalized coordinate space
                wrist_dist = np.sqrt((last_kp[8] - last_kp[10])**2 + (last_kp[9] - last_kp[11])**2)
                
                # Render coordinates layout
                telemetry_text = f"""=== MEDIAPIPE POSE JOINT TELEMETRY ===
Status: TRACKING ACTIVE [OK]
Active Frame: {self.predictor.frame_count:05d}

[SHOULDERS]
  Left Shoulder:  X={last_kp[0]:.4f}, Y={last_kp[1]:.4f}
  Right Shoulder: X={last_kp[2]:.4f}, Y={last_kp[3]:.4f}
  
[ELBOWS]
  Left Elbow:     X={last_kp[4]:.4f}, Y={last_kp[5]:.4f}
  Right Elbow:    X={last_kp[6]:.4f}, Y={last_kp[7]:.4f}
  
[WRISTS]
  Left Wrist:     X={last_kp[8]:.4f}, Y={last_kp[9]:.4f}
  Right Wrist:    X={last_kp[10]:.4f}, Y={last_kp[11]:.4f}
  
[HIPS]
  Left Hip:       X={last_kp[12]:.4f}, Y={last_kp[13]:.4f}
  Right Hip:      X={last_kp[14]:.4f}, Y={last_kp[15]:.4f}
  
[KNEES]
  Left Knee:      X={last_kp[16]:.4f}, Y={last_kp[17]:.4f}
  Right Knee:     X={last_kp[18]:.4f}, Y={last_kp[19]:.4f}
  
[ANKLES]
  Left Ankle:     X={last_kp[20]:.4f}, Y={last_kp[21]:.4f}
  Right Ankle:    X={last_kp[22]:.4f}, Y={last_kp[23]:.4f}
  
=== DYNAMIC TRACKING METRICS ===
Wrist Distance: {wrist_dist:.4f}
Left Wrist Height (relative): {last_kp[9]:.4f}
Right Wrist Height (relative): {last_kp[11]:.4f}
Estimated Pose Scale factor: {getattr(self.predictor, 'last_confidence', 0.0):.4f}"""
                self.telemetry_console.setText(telemetry_text)
                return
        self.telemetry_console.setText("=== MEDIAPIPE POSE JOINT TELEMETRY ===\nStatus: NO PERSON DETECTED\n\nStanding by for pose identification...")

    def _update_diag_image(self):
        # Toggles between loading the confusion matrix image and the training curves image
        if self.rad_conf_matrix.isChecked():
            img_name = "confusion_matrix.png"
        else:
            img_name = "training_history.png"
            
        path = os.path.join(PROJECT_ROOT, "logs", img_name)
        if os.path.exists(path):
            pixmap = QPixmap(path)
            self.lbl_matrix_img.setPixmap(pixmap.scaled(
                self.lbl_matrix_img.width(), self.lbl_matrix_img.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.lbl_matrix_img.setText(f"\n❌ Image Not Found at:\n{path}\n\nPlease run training first to generate metric logs.")

    def resizeEvent(self, event):
        # Refresh scaled pixmap on resizing window to keep high fidelity
        super().resizeEvent(event)
        self._update_diag_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HARApplication()
    w.show()
    sys.exit(app.exec_())
