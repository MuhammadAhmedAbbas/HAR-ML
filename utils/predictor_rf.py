"""
utils/predictor_rf.py
======================
Random Forest predictor - NO TensorFlow, NO DLL issues!

Pure scikit-learn inference with:
  [OK] Consistent features
  [OK] Confidence thresholding
  [OK] Prediction smoothing
"""

import os
import sys
import cv2
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# Match training constants
SEQUENCE_LENGTH = 20
RELEVANT_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NUM_FEATURES = len(RELEVANT_LANDMARKS) * 2
CONFIDENCE_THRESHOLD = 0.35
ACTIVITY_CLASSES = ["handclapping", "handwaving", "sitting", "standing", "walking"]

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

def _classify_static_pose(unscaled_buffer):
    """
    When the RF model is uncertain, use raw normalized joint geometry to
    classify whether the person is sitting or standing.

    Uses a SCORING SYSTEM over two independent measurements so a single
    noisy frame cannot flip the label:
      - Measurement 1: knee-hip vertical gap   (primary signal)
      - Measurement 2: ankle-knee vertical gap (secondary signal)

    Landmark Y-axis mapping (torso-normalised, origin = shoulder midpoint):
      L-Hip   → idx 13   R-Hip   → idx 15
      L-Knee  → idx 17   R-Knee  → idx 19
      L-Ankle → idx 21   R-Ankle → idx 23

    Standing: thigh hangs downward  → knee well below hip AND ankle well below knee.
    Sitting:  thigh roughly horizontal → knee near same Y as hip, ankle may vary.
    """
    valid_frames = [kp for kp in unscaled_buffer if np.sum(kp) != 0]
    if not valid_frames:
        return "standing"  # Default safe fallback

    # Use MEDIAN (not mean) so single-frame outliers don't distort the result
    ref_kp = np.median(valid_frames, axis=0)  # shape: (24,)

    # --- Motion variance: if joints are moving a lot keep as Uncertain ----------
    motion_std = np.std(valid_frames, axis=0).mean()
    if motion_std > 0.15:
        return "Uncertain"

    # --- Extract vertical positions (Y increases downward in image) ------------
    mid_hip_y   = (ref_kp[13] + ref_kp[15]) / 2.0   # avg of L-Hip Y & R-Hip Y
    mid_knee_y  = (ref_kp[17] + ref_kp[19]) / 2.0   # avg of L-Knee Y & R-Knee Y
    mid_ankle_y = (ref_kp[21] + ref_kp[23]) / 2.0   # avg of L-Ankle Y & R-Ankle Y

    knee_hip_diff   = mid_knee_y  - mid_hip_y    # large +ve → knees far below hips (standing)
    ankle_knee_diff = mid_ankle_y - mid_knee_y   # large +ve → ankles far below knees (standing)

    # --- Score-based decision --------------------------------------------------
    # Each criterion awards 0-2 points toward "standing". Total max = 4.
    # Score >= 3 → standing
    # Score <= 1 → sitting
    # Score == 2 → ambiguous, use knee-hip as tiebreaker
    score = 0

    # Criterion 1: knee-hip gap (strong discriminator)
    if knee_hip_diff > 0.55:
        score += 2   # Clearly standing: knees well below hips
    elif knee_hip_diff > 0.30:
        score += 1   # Borderline

    # Criterion 2: ankle-knee gap (confirms leg is fully extended)
    if ankle_knee_diff > 0.45:
        score += 2   # Clearly standing: ankles well below knees
    elif ankle_knee_diff > 0.20:
        score += 1   # Borderline

    if score >= 3:
        return "standing"
    elif score <= 1:
        return "sitting"
    else:
        # Ambiguous (score == 2): use the stronger individual signal
        return "standing" if knee_hip_diff > 0.40 else "sitting"


def _init_mediapipe_pose():
    import mediapipe as mp
    mp_module = mp
    mp_api = "new"
    
    model_path = os.path.join(MODELS_PATH, "pose_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        print("  Downloading MediaPipe Pose model...")
        os.makedirs(MODELS_PATH, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
    
    mp_tasks = mp.tasks
    mp_vision = mp_tasks.vision
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=mp_vision.RunningMode.IMAGE
    )
    return mp_vision.PoseLandmarker.create_from_options(options), mp_api, mp_module

class ActivityPredictor:
    def __init__(self):
        self.buffer = []
        self.unscaled_buffer = []
        self.frame_count = 0
        self.last_label = "Initializing..."
        self.last_confidence = 0.0
        self.last_probabilities = np.zeros(len(ACTIVITY_CLASSES))
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.pose_model = None

        self._load_encoders()
        self._load_model()
        self._init_mediapipe()

    def _load_encoders(self):
        le_path = os.path.join(MODELS_PATH, "label_encoder.pkl")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
        
        scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def _load_model(self):
        rf_path = os.path.join(MODELS_PATH, "har_rf_model.pkl")
        if os.path.exists(rf_path):
            import pickle
            with open(rf_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model not found at {rf_path}")

    def _init_mediapipe(self):
        self.pose_model, self.mp_api, self.mp_module = _init_mediapipe_pose()

    def predict_frame(self, bgr_frame):
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        annotated_frame = bgr_frame.copy()
        pose_detected = False
        keypoints = None

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.pose_model.detect(mp_image)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose_detected = True
            kp = []
            for i in RELEVANT_LANDMARKS:
                lm = results.pose_landmarks[0][i]
                kp.extend([lm.x, lm.y])
            keypoints = np.array(kp, dtype=np.float32)
            
            h, w = annotated_frame.shape[:2]
            for lm in results.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 128), -1)

        if self.frame_count % 2 == 0:
            if keypoints is not None:
                norm_kp = normalize_pose(keypoints)
                self.unscaled_buffer.append(norm_kp)
                if self.scaler is not None:
                    scaled_kp = self.scaler.transform(norm_kp.reshape(1, -1))[0]
                else:
                    scaled_kp = norm_kp
                self.buffer.append(scaled_kp)
            else:
                self.buffer.append(np.zeros(NUM_FEATURES, dtype=np.float32))
                self.unscaled_buffer.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            if len(self.buffer) > SEQUENCE_LENGTH:
                self.buffer.pop(0)
                self.unscaled_buffer.pop(0)

        if len(self.buffer) == SEQUENCE_LENGTH:
            seq = np.array(self.buffer, dtype=np.float32)
            X_mean = seq.mean(axis=0, keepdims=True)
            X_std  = seq.std(axis=0, keepdims=True)
            X_min  = seq.min(axis=0, keepdims=True)
            X_max  = seq.max(axis=0, keepdims=True)
            X_in   = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)

            pred_idx = self.model.predict(X_in)[0]
            proba = self.model.predict_proba(X_in)[0]
            confidence = float(proba[pred_idx])

            if self.label_encoder is not None:
                label = self.label_encoder.classes_[pred_idx]
            else:
                label = ACTIVITY_CLASSES[pred_idx]

            # --- Body-part-specific motion guards ---------------------------------
            # Using whole-body std is too coarse. Each activity type only produces
            # motion in specific joints. We compute per-region variance to avoid
            # suppressing genuine gestures while still blocking false positives.
            #
            # RELEVANT_LANDMARKS order = [11,12,13,14,15,16,23,24,25,26,27,28]
            # In the 24-feature (x,y) array:
            #   indices 0-3   -> Shoulders (11,12)   kp[0..3]
            #   indices 4-7   -> Elbows    (13,14)   kp[4..7]
            #   indices 8-11  -> Wrists    (15,16)   kp[8..11]
            #   indices 12-15 -> Hips      (23,24)   kp[12..15]
            #   indices 16-19 -> Knees     (25,26)   kp[16..19]
            #   indices 20-23 -> Ankles    (27,28)   kp[20..23]
            valid_frames = [kp for kp in self.unscaled_buffer if np.sum(kp) != 0]
            if len(valid_frames) > 1:
                fa = np.array(valid_frames, dtype=np.float32)
                # Wrist motion: only wrists moving -> hand gestures
                wrist_std = fa[:, 8:12].std(axis=0).mean()
                # Leg motion:  knees + ankles moving -> walking/running/jogging
                leg_std   = fa[:, 16:24].std(axis=0).mean()
                # Arm motion:  elbows + wrists moving -> boxing / hand gestures
                arm_std   = fa[:, 4:12].std(axis=0).mean()
            else:
                wrist_std = leg_std = arm_std = 0.0

            # --- Guard: Hand gestures require genuine wrist movement -----------
            # Threshold 0.04: wrists must travel noticeably to justify the label.
            if label in ["handclapping", "handwaving"]:
                if wrist_std < 0.04:
                    # Wrists barely moving -> this is a false positive
                    label = _classify_static_pose(self.unscaled_buffer)
                else:
                    # Wrists ARE moving -> run the clapping vs waving disambiguator
                    min_wrist_dist  = float('inf')
                    max_wrist_height = float('inf')   # smaller Y = higher in image
                    for kp in self.unscaled_buffer:
                        if np.sum(kp) == 0: continue
                        dist = np.sqrt((kp[8] - kp[10])**2 + (kp[9] - kp[11])**2)
                        if dist < min_wrist_dist:
                            min_wrist_dist = dist
                        highest_y = min(kp[9], kp[11])
                        if highest_y < max_wrist_height:
                            max_wrist_height = highest_y
                    # Clapping: wrists come close + stay near chest
                    if min_wrist_dist < 0.35 and max_wrist_height > -0.2:
                        label = "handclapping"
                    # Waving: hand rises above shoulder line or wrists stay far apart
                    elif max_wrist_height < -0.1 or min_wrist_dist > 0.5:
                        label = "handwaving"

            # --- Guard: Walking requires leg movement -------------------------
            elif label == "walking" and leg_std < 0.04:
                label = _classify_static_pose(self.unscaled_buffer)

            # --- Guard: sitting/standing from model confirmed by geometry -----
            elif label in ["sitting", "standing"]:
                # Let the geometry heuristic override if it strongly disagrees
                geo_label = _classify_static_pose(self.unscaled_buffer)
                if geo_label not in ["Uncertain"] and geo_label != "standing" and label == "standing":
                    label = geo_label
                elif geo_label == "standing" and label == "sitting":
                    label = "standing"

            if confidence < CONFIDENCE_THRESHOLD:
                # Instead of a plain "Uncertain", classify static pose
                label = _classify_static_pose(self.unscaled_buffer)

            self.last_label = label
            self.last_confidence = confidence
            self.last_probabilities = proba
        elif not pose_detected:
            self.last_probabilities = np.zeros(len(ACTIVITY_CLASSES))
            return "No Person Detected", 0.0, annotated_frame, False
        else:
            self.last_probabilities = np.zeros(len(ACTIVITY_CLASSES))
            return f"Buffering... ({len(self.buffer)}/{SEQUENCE_LENGTH})", 0.0, annotated_frame, True

        return self.last_label, self.last_confidence, annotated_frame, pose_detected

    def reset_buffer(self):
        self.buffer = []
        self.unscaled_buffer = []
        self.frame_count = 0
        self.last_label = "Initializing..."
        self.last_confidence = 0.0
        self.last_probabilities = np.zeros(len(ACTIVITY_CLASSES))
