"""
utils/predictor.py
====================
ActivityPredictor -- the inference engine for real-time HAR.

Handles both legacy MediaPipe API (< 0.10) and the new Tasks API (>= 0.10).
"""

import os
import sys
import numpy as np
import joblib
import cv2

# ---- Configuration -----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH  = os.path.join(PROJECT_ROOT, "models")

SEQUENCE_LENGTH  = 20
RELEVANT_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NUM_FEATURES     = len(RELEVANT_LANDMARKS) * 2    # 12 landmarks x 2 values (x,y)
NUM_LANDMARKS    = 12
ACTIVITY_CLASSES = ["handclapping", "handwaving", "sitting", "standing", "walking"]
CONFIDENCE_THRESHOLD = 0.40

def normalize_pose(kp):
    """Make pose translation and scale invariant using 12 core landmarks."""
    if np.sum(kp) == 0: return kp
    
    # 0: L-Shoulder, 1: R-Shoulder, 6: L-Hip, 7: R-Hip
    ls_x, ls_y = kp[0*2], kp[0*2 + 1]
    rs_x, rs_y = kp[1*2], kp[1*2 + 1]
    lh_x, lh_y = kp[6*2], kp[6*2 + 1]
    rh_x, rh_y = kp[7*2], kp[7*2 + 1]
    
    mid_shoulder_x = (ls_x + rs_x) / 2
    mid_shoulder_y = (ls_y + rs_y) / 2
    mid_hip_x = (lh_x + rh_x) / 2
    mid_hip_y = (lh_y + rh_y) / 2
    
    torso = np.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
    if torso < 0.01: torso = 0.01
        
    norm_kp = np.zeros_like(kp)
    for i in range(len(RELEVANT_LANDMARKS)):
        idx = i * 2
        norm_kp[idx]   = (kp[idx] - mid_shoulder_x) / torso
        norm_kp[idx+1] = (kp[idx+1] - mid_shoulder_y) / torso
    return norm_kp


def _init_mediapipe_pose():
    """
    Initialize MediaPipe Pose, supporting both old API (mp.solutions.pose)
    and new Tasks API (mediapipe >= 0.10).

    Returns
    -------
    pose_obj  : pose model instance
    api       : "legacy" or "new"
    mp_module : mediapipe module (for drawing utilities when using legacy API)
    """
    import mediapipe as mp

    # Try legacy API first
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return pose, "legacy", mp
    except Exception:
        pass

    # Fall back to new Tasks API
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = os.path.join(MODELS_PATH, "pose_landmarker.task")
        if not os.path.exists(model_path):
            print("  Downloading MediaPipe Pose Landmarker model (~5 MB)...")
            import urllib.request
            os.makedirs(MODELS_PATH, exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/1/"
                   "pose_landmarker_lite.task")
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded.")

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE   # IMAGE mode - no timestamps needed
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        print("  MediaPipe Tasks API initialized (IMAGE mode).")
        return detector, "new", mp
    except Exception as e:
        raise RuntimeError(f"Cannot initialize MediaPipe Pose: {e}")


class ActivityPredictor:
    """
    End-to-end inference pipeline for real-time activity recognition.
    Maintains a rolling 30-frame buffer and returns (label, confidence).
    """

    def __init__(self):
        self.model         = None
        self.model_type    = None
        self.label_encoder = None
        self.scaler        = None
        self.pose_model    = None
        self.mp_api        = None
        self.mp_module     = None

        self.buffer          = []
        self.last_label      = "Initializing..."
        self.last_confidence = 0.0
        self.frame_count     = 0

        self._load_label_encoder()
        self._load_scaler()
        self._load_model()
        self._init_mediapipe()

    # ---- Private init --------------------------------------------------------

    def _load_label_encoder(self):
        le_path = os.path.join(MODELS_PATH, "label_encoder.pkl")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
        else:
            print("  [WARNING] label_encoder.pkl not found. Using defaults.")

    def _load_scaler(self):
        scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print("  [WARNING] scaler.pkl not found.")

    def _load_model(self):
        rf_path = os.path.join(MODELS_PATH, "har_rf_model.pkl")

        if os.path.exists(rf_path):
            print("  Loading Random Forest model...")
            import pickle
            with open(rf_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = "rf"
            print("  [OK] Random Forest model loaded")
        else:
            raise FileNotFoundError(
                f"No trained model found in {MODELS_PATH}\n"
                "  Train a model first:\n"
                "    py -3 training/train_random_forest.py"
            )

    def _init_mediapipe(self):
        self.pose_model, self.mp_api, self.mp_module = _init_mediapipe_pose()

        # Drawing utilities only available in legacy API
        if self.mp_api == "legacy":
            self.mp_drawing        = self.mp_module.solutions.drawing_utils
            self.mp_drawing_styles = self.mp_module.solutions.drawing_styles
            self.mp_pose_conns     = self.mp_module.solutions.pose.POSE_CONNECTIONS
        else:
            self.mp_drawing    = None
            self.mp_pose_conns = None

        print(f"  [OK] MediaPipe Pose initialized (API: {self.mp_api})")

    # ---- Public API ----------------------------------------------------------

    def predict_frame(self, bgr_frame):
        """
        Process one BGR frame. Returns (label, confidence, annotated_frame, pose_detected).
        """
        self.frame_count  += 1

        rgb_frame       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        annotated_frame = bgr_frame.copy()
        pose_detected   = False
        keypoints       = None

        # ---- Run pose estimation -----------------------------------------
        if self.mp_api == "legacy":
            results = self.pose_model.process(rgb_frame)

            if results.pose_landmarks:
                pose_detected = True
                keypoints = self._extract_keypoints_legacy(results)

                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose_conns,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 128), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 200, 255), thickness=2
                    )
                )
        else:
            # New Tasks API
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results  = self.pose_model.detect(mp_image)   # IMAGE mode: no timestamp needed

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_detected = True
                keypoints = self._extract_keypoints_new(results)
                # Draw simple dots for new API
                annotated_frame = self._draw_landmarks_new(
                    annotated_frame, results.pose_world_landmarks
                    if results.pose_world_landmarks else results.pose_landmarks
                )

        # ---- Update buffer -----------------------------------------------
        # Match training FRAME_SKIP=2 (only add every 2nd frame to buffer)
        if self.frame_count % 2 == 0:
            if keypoints is not None:
                keypoints = normalize_pose(keypoints)
                if self.scaler is not None:
                    keypoints = self.scaler.transform(keypoints.reshape(1, -1))[0]
                self.buffer.append(keypoints)
            else:
                self.buffer.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            if len(self.buffer) > SEQUENCE_LENGTH:
                self.buffer.pop(0)

        # ---- Predict --------------------------------------------------------
        if len(self.buffer) == SEQUENCE_LENGTH:
            label, confidence         = self._run_prediction()
            self.last_label           = label
            self.last_confidence      = confidence
        elif not pose_detected:
            return "No Person Detected", 0.0, annotated_frame, False
        else:
            progress = len(self.buffer)
            return f"Buffering... ({progress}/{SEQUENCE_LENGTH})", 0.0, annotated_frame, True

        return self.last_label, self.last_confidence, annotated_frame, pose_detected

    # ---- Keypoint extraction ------------------------------------------------

    def _extract_keypoints_legacy(self, results):
        kp = []
        for i in RELEVANT_LANDMARKS:
            lm = results.pose_landmarks.landmark[i]
            kp.extend([lm.x, lm.y])
        return np.array(kp, dtype=np.float32)

    def _extract_keypoints_new(self, results):
        kp = []
        for i in RELEVANT_LANDMARKS:
            lm = results.pose_landmarks[0][i]
            kp.extend([lm.x, lm.y])
        return np.array(kp, dtype=np.float32)

    def _draw_landmarks_new(self, frame, landmarks_list):
        """Simple dot drawing for new Tasks API (no drawing_utils)."""
        if not landmarks_list:
            return frame
        h, w = frame.shape[:2]
        for lm in landmarks_list[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 128), -1)
        return frame

    # ---- Inference ----------------------------------------------------------

    def _run_prediction(self):
        sequence = np.array(self.buffer, dtype=np.float32)

        if self.model_type == "lstm":
            X     = sequence[np.newaxis, ...]
            proba = self.model.predict(X, verbose=0)[0]
            pred_idx   = np.argmax(proba)
            confidence = float(proba[pred_idx])
        else:
            X_mean = sequence.mean(axis=0, keepdims=True)
            X_std  = sequence.std(axis=0, keepdims=True)
            X_min  = sequence.min(axis=0, keepdims=True)
            X_max  = sequence.max(axis=0, keepdims=True)
            # EXCLUDE X_flat to match the new time-invariant model
            X_in   = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)

            pred_idx   = self.model.predict(X_in)[0]
            proba      = self.model.predict_proba(X_in)[0]
            confidence = float(proba[pred_idx])

        if self.label_encoder is not None:
            label = self.label_encoder.classes_[pred_idx]
        else:
            label = ACTIVITY_CLASSES[pred_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return "Uncertain", confidence

        return label, confidence

    # ---- Utilities ----------------------------------------------------------

    def get_model_info(self):
        return {
            "type"         : self.model_type,
            "display_name" : "LSTM (Deep Learning)" if self.model_type == "lstm" else "Random Forest",
            "buffer_size"  : len(self.buffer),
            "frame_count"  : self.frame_count,
        }

    def reset_buffer(self):
        self.buffer          = []
        self.last_label      = "Initializing..."
        self.last_confidence = 0.0
        self.frame_count     = 0
        # Note: do NOT reset _timestamp_ms — it must always increase for video mode

    def release(self):
        if self.pose_model:
            try:
                self.pose_model.close()
            except Exception:
                pass
