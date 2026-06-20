"""
quick_train.py
================
FAST setup script — creates a working model in ~3-5 minutes.

This script:
  1. Picks 8 videos per class from the KTH dataset
  2. Extracts MediaPipe keypoints directly (no saved .npy files needed)
  3. Trains a Random Forest classifier
  4. Saves the model so the GUI works immediately

Run this ONCE to get the app working:
    py -3 quick_train.py

Then launch the GUI:
    py -3 gui/app.py
"""

import os
import sys
import cv2
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Fix Unicode on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---- Configuration -------------------------------------------------------

PROJECT_ROOT     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH     = os.path.join(PROJECT_ROOT, "KTH dataset")
MODELS_PATH      = os.path.join(PROJECT_ROOT, "models")

ACTIVITY_CLASSES = ["handclapping", "handwaving", "sitting", "standing", "walking"]
VIDEOS_PER_CLASS = 15       # More videos
FRAME_SKIP       = 2
SEQUENCE_LENGTH  = 20
STRIDE           = 5
RELEVANT_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FEATURES         = len(RELEVANT_LANDMARKS) * 2  # 24

os.makedirs(MODELS_PATH, exist_ok=True)


# ---- MediaPipe setup -----------------------------------------------------

def init_mediapipe():
    """Initialize MediaPipe Pose (handles new Tasks API for v0.10+)."""
    import mediapipe as mp

    # New Tasks API (mediapipe >= 0.10)
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = os.path.join(MODELS_PATH, "pose_landmarker.task")
        if not os.path.exists(model_path):
            print("  Downloading MediaPipe pose model (~5 MB)...")
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/1/"
                   "pose_landmarker_lite.task")
            urllib.request.urlretrieve(url, model_path)
            print("  Downloaded.")

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE   # IMAGE mode — simpler, no timestamps
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        print("  MediaPipe Tasks API initialized.")
        return detector, "new", mp

    except Exception as e:
        print(f"  Tasks API failed ({e}), trying legacy API...")

    # Legacy API (mediapipe < 0.10)
    try:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,   # Lite model — faster
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        print("  MediaPipe legacy API initialized.")
        return pose, "legacy", mp
    except Exception as e2:
        raise RuntimeError(f"Cannot init MediaPipe: {e2}")


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

def extract_frame_keypoints(rgb_frame, pose_model, api, mp_mod):
    """Extract X, Y for 12 core landmarks."""
    try:
        if api == "new":
            img = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb_frame)
            results = pose_model.detect(img)
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                kp = []
                for i in RELEVANT_LANDMARKS:
                    lm = results.pose_landmarks[0][i]
                    kp.extend([lm.x, lm.y])
                return normalize_pose(np.array(kp, dtype=np.float32))
        else:
            results = pose_model.process(rgb_frame)
            if results.pose_landmarks:
                kp = []
                for i in RELEVANT_LANDMARKS:
                    lm = results.pose_landmarks.landmark[i]
                    kp.extend([lm.x, lm.y])
                return normalize_pose(np.array(kp, dtype=np.float32))
    except Exception:
        pass
    return np.zeros(FEATURES, dtype=np.float32)


def process_video(video_path, pose_model, api, mp_mod):
    """Extract keypoints from a video. Returns array of shape (T, 132) or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    all_kp = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp  = extract_frame_keypoints(rgb, pose_model, api, mp_mod)
            all_kp.append(kp)
        idx += 1

    cap.release()
    return np.array(all_kp, dtype=np.float32) if all_kp else None


def make_windows(sequence, label):
    """Slice sequence into overlapping windows of SEQUENCE_LENGTH frames."""
    X, y = [], []
    for start in range(0, len(sequence) - SEQUENCE_LENGTH + 1, STRIDE):
        window = sequence[start:start + SEQUENCE_LENGTH]
        X.append(window)
        y.append(label)
    return X, y


# ---- Main ----------------------------------------------------------------

# ---- Synthetic data generators for Sitting and Standing --------------------

KTH_CLASSES = ["handclapping", "handwaving", "walking"]   # classes present in KTH dataset
SYNTH_CLASSES = ["sitting", "standing"]                   # classes generated synthetically


def _make_standing_frame(noise=0.02, rng=None):
    """Generate a single normalized pose frame for a standing person.
    Landmark order: [L-Shoulder, R-Shoulder, L-Elbow, R-Elbow, L-Wrist, R-Wrist,
                     L-Hip, R-Hip, L-Knee, R-Knee, L-Ankle, R-Ankle]
    Coordinates are torso-normalised (origin = shoulder midpoint, unit = torso length).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = lambda s=noise: rng.normal(0, s)

    kp = np.array([
        # L-Shoulder        R-Shoulder
        -0.25 + n(), 0.0 + n(),   0.25 + n(), 0.0 + n(),
        # L-Elbow           R-Elbow
        -0.35 + n(), 0.40 + n(),  0.35 + n(), 0.40 + n(),
        # L-Wrist           R-Wrist
        -0.30 + n(), 0.75 + n(),  0.30 + n(), 0.75 + n(),
        # L-Hip             R-Hip
        -0.20 + n(), 0.95 + n(),  0.20 + n(), 0.95 + n(),
        # L-Knee            R-Knee  (well below hips → standing)
        -0.18 + n(), 1.70 + n(),  0.18 + n(), 1.70 + n(),
        # L-Ankle           R-Ankle
        -0.16 + n(), 2.40 + n(),  0.16 + n(), 2.40 + n(),
    ], dtype=np.float32)
    return kp


def _make_sitting_frame(noise=0.02, rng=None):
    """Generate a single normalised pose frame for a sitting person.
    In a seated posture the thigh is roughly horizontal so knee Y ≈ hip Y;
    the lower leg hangs down so ankle Y > knee Y, but not by as much as standing.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = lambda s=noise: rng.normal(0, s)

    kp = np.array([
        # L-Shoulder        R-Shoulder
        -0.25 + n(), 0.0 + n(),   0.25 + n(), 0.0 + n(),
        # L-Elbow           R-Elbow
        -0.35 + n(), 0.40 + n(),  0.35 + n(), 0.40 + n(),
        # L-Wrist           R-Wrist (hands often resting on lap)
        -0.20 + n(), 0.85 + n(),  0.20 + n(), 0.85 + n(),
        # L-Hip             R-Hip
        -0.20 + n(), 0.95 + n(),  0.20 + n(), 0.95 + n(),
        # L-Knee            R-Knee  (near same Y as hips → seated thigh)
        -0.18 + n(), 1.05 + n(),  0.18 + n(), 1.05 + n(),
        # L-Ankle           R-Ankle (hanging ~below knee)
        -0.16 + n(), 1.60 + n(),  0.16 + n(), 1.60 + n(),
    ], dtype=np.float32)
    return kp


def generate_synthetic_sequences(activity, n_sequences=120, rng=None):
    """Build a list of synthetic frame sequences for 'sitting' or 'standing'.
    Each sequence has SEQUENCE_LENGTH frames with small per-sequence pose variation
    (person position) plus per-frame noise (natural micro-movements).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    gen_fn = _make_sitting_frame if activity == "sitting" else _make_standing_frame
    sequences = []
    for _ in range(n_sequences):
        # Small constant per-sequence offset to simulate different people / distances
        seq_offset = rng.normal(0, 0.04, size=FEATURES).astype(np.float32)
        frames = []
        for _ in range(SEQUENCE_LENGTH):
            frame = gen_fn(noise=0.025, rng=rng) + seq_offset
            frames.append(frame)
        sequences.append(np.array(frames, dtype=np.float32))
    return sequences


# ---- Main -------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  HAR Quick-Train Script")
    print(f"  Classes: {ACTIVITY_CLASSES}")
    print("  Expected time: ~3-5 minutes")
    print("=" * 60)

    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] KTH dataset not found at: {DATASET_PATH}")
        sys.exit(1)

    # Init MediaPipe
    print("\n[1/4] Initializing MediaPipe Pose...")
    pose_model, api, mp_mod = init_mediapipe()

    # Extract keypoints from KTH videos
    print(f"\n[2/4] Extracting keypoints from KTH dataset ({VIDEOS_PER_CLASS} videos per class)...")
    all_sequences = []
    all_labels    = []

    for activity in KTH_CLASSES:
        label_idx = ACTIVITY_CLASSES.index(activity)
        folder = os.path.join(DATASET_PATH, activity)
        if not os.path.exists(folder):
            print(f"  [SKIP] folder not found: {folder}")
            continue
        videos = sorted([f for f in os.listdir(folder) if f.endswith(".avi")])

        step     = max(1, len(videos) // VIDEOS_PER_CLASS)
        selected = videos[::step][:VIDEOS_PER_CLASS]

        print(f"  [{activity}] processing {len(selected)} videos...", end="", flush=True)
        ok = 0
        for vfile in selected:
            vpath = os.path.join(folder, vfile)
            seq   = process_video(vpath, pose_model, api, mp_mod)
            if seq is not None and len(seq) >= SEQUENCE_LENGTH:
                all_sequences.append(seq)
                all_labels.append(label_idx)
                ok += 1
        print(f" done ({ok} ok)")

    # Generate synthetic sitting / standing sequences
    print("\n  Generating synthetic Sitting and Standing sequences...")
    rng = np.random.default_rng(42)
    SYNTH_N = VIDEOS_PER_CLASS * 8   # enough sequences to balance KTH classes
    for activity in SYNTH_CLASSES:
        label_idx = ACTIVITY_CLASSES.index(activity)
        synth_seqs = generate_synthetic_sequences(activity, n_sequences=SYNTH_N, rng=rng)
        for seq in synth_seqs:
            all_sequences.append(seq)
            all_labels.append(label_idx)
        print(f"  [{activity}] generated {SYNTH_N} synthetic sequences")

    print(f"\n  Total sequences collected: {len(all_sequences)}")

    if len(all_sequences) == 0:
        print("[ERROR] No valid sequences extracted. Check your dataset path.")
        sys.exit(1)

    # Build windowed dataset
    print("\n[3/4] Building training dataset...")
    X_list, y_list = [], []
    for seq, lbl in zip(all_sequences, all_labels):
        Xw, yw = make_windows(seq, lbl)
        X_list.extend(Xw)
        y_list.extend(yw)

    X_seq = np.array(X_list, dtype=np.float32)   # (N, 30, 132)
    y     = np.array(y_list,  dtype=np.int32)

    print(f"  Windows: {X_seq.shape[0]} total")

    # Normalize
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    all_frames = X_seq.reshape(-1, FEATURES)
    scaler = StandardScaler()
    scaler.fit(all_frames)

    X_norm = np.array([scaler.transform(w) for w in X_seq])

    # Add statistical features (mean, std, min, max per window)
    # We purposefully EXCLUDE X_flat to make the model time-invariant (ignores phase)
    X_mean = X_norm.mean(axis=1)
    X_std  = X_norm.std(axis=1)
    X_min  = X_norm.min(axis=1)
    X_max  = X_norm.max(axis=1)
    X_feat = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)

    print(f"  Feature shape: {X_feat.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    print("\n[4/4] Training Random Forest...")
    print("  (This takes ~1-3 minutes...)")

    rf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        verbose=0
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred   = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy*100:.1f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=ACTIVITY_CLASSES))

    # Save model artifacts
    le = LabelEncoder()
    le.fit(ACTIVITY_CLASSES)

    joblib.dump(rf,     os.path.join(MODELS_PATH, "har_rf_model.pkl"))
    joblib.dump(le,     os.path.join(MODELS_PATH, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

    # Save test data for evaluate_model.py
    os.makedirs(os.path.join(PROJECT_ROOT, "dataset"), exist_ok=True)
    np.save(os.path.join(PROJECT_ROOT, "dataset", "X_test.npy"), X_test)
    np.save(os.path.join(PROJECT_ROOT, "dataset", "y_test.npy"), y_test)

    # Cleanup MediaPipe
    try:
        pose_model.close()
    except Exception:
        pass

    print()
    print("=" * 60)
    print("  [DONE] Model saved!")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Files saved to: {MODELS_PATH}")
    print()
    print("  Launch the GUI now:")
    print("    py -3 gui/app.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
