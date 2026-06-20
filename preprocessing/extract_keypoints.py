"""
preprocessing/extract_keypoints.py
====================================
Step 1 of the HAR preprocessing pipeline.

PURPOSE:
    Read every video from the KTH dataset, extract MediaPipe Pose landmarks
    from each frame, and save the per-video keypoint sequence as a .npy file.

OUTPUT:
    dataset/keypoints/<activity>/<video_name>.npy
    Each file contains an array of shape (num_frames, 132)
    132 = 33 landmarks x (x, y, z, visibility)

USAGE:
    py -3 preprocessing/extract_keypoints.py
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Fix Unicode output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---- Configuration -------------------------------------------------------

# Root of the project (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the raw KTH dataset videos
DATASET_PATH = os.path.join(PROJECT_ROOT, "KTH dataset")

# Where to save extracted keypoint .npy files
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "keypoints")

# The 6 activity classes (folder names inside DATASET_PATH)
ACTIVITY_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# Extract every Nth frame (3 = skip 2 frames between each extracted frame)
FRAME_SKIP = 3

# 33 landmarks x 4 values (x, y, z, visibility) = 132 features
NUM_LANDMARKS = 33
NUM_FEATURES_PER_LANDMARK = 4
FEATURES_PER_FRAME = NUM_LANDMARKS * NUM_FEATURES_PER_LANDMARK  # 132


def create_pose_model():
    """
    Create and return a MediaPipe Pose model.
    Handles both legacy API (< 0.10) and new API (>= 0.10).
    """
    try:
        # Try legacy API first (mediapipe < 0.10)
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return pose, "legacy"
    except AttributeError:
        pass

    try:
        # New API (mediapipe >= 0.10)
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        # Download model if not present
        model_path = os.path.join(PROJECT_ROOT, "models", "pose_landmarker.task")
        if not os.path.exists(model_path):
            print("  Downloading MediaPipe Pose model...")
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded.")

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        return detector, "new"
    except Exception as e:
        raise RuntimeError(f"Cannot initialize MediaPipe Pose: {e}")


def extract_keypoints_legacy(results):
    """
    Extract keypoints from legacy MediaPipe Pose results.
    Returns np.ndarray of shape (132,)
    """
    if results.pose_landmarks:
        kp = []
        for landmark in results.pose_landmarks.landmark:
            kp.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(kp, dtype=np.float32)
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def extract_keypoints_new(results):
    """
    Extract keypoints from new MediaPipe Tasks API results.
    Returns np.ndarray of shape (132,)
    """
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        kp = []
        for landmark in results.pose_landmarks[0]:
            kp.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(kp, dtype=np.float32)
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def extract_video_keypoints(video_path, pose_model, api_version):
    """
    Extract pose keypoints from every Nth frame of a single video.

    Parameters
    ----------
    video_path  : str  -- full path to the .avi video file
    pose_model  -- MediaPipe Pose model object
    api_version : str  -- "legacy" or "new"

    Returns
    -------
    np.ndarray of shape (num_extracted_frames, 132) or None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    all_keypoints = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_SKIP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if api_version == "legacy":
                results = pose_model.process(rgb_frame)
                kp = extract_keypoints_legacy(results)
            else:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = pose_model.detect(mp_image)
                kp = extract_keypoints_new(results)

            all_keypoints.append(kp)

        frame_index += 1

    cap.release()

    if len(all_keypoints) == 0:
        return None

    return np.array(all_keypoints, dtype=np.float32)


def process_all_videos():
    """
    Main function: iterate over all activities and videos,
    extract keypoints, and save them as .npy files.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for activity in ACTIVITY_CLASSES:
        os.makedirs(os.path.join(OUTPUT_PATH, activity), exist_ok=True)

    print("  Initializing MediaPipe Pose model...")
    pose_model, api_version = create_pose_model()
    print(f"  Using MediaPipe API: {api_version}")

    total_processed = 0
    total_skipped = 0

    print("=" * 60)
    print("  HAR -- MediaPipe Keypoint Extraction")
    print("=" * 60)
    print(f"  Dataset path : {DATASET_PATH}")
    print(f"  Output path  : {OUTPUT_PATH}")
    print(f"  Frame skip   : every {FRAME_SKIP} frames")
    print("=" * 60)

    for activity in ACTIVITY_CLASSES:
        activity_folder = os.path.join(DATASET_PATH, activity)
        output_folder   = os.path.join(OUTPUT_PATH, activity)

        video_files = sorted([f for f in os.listdir(activity_folder) if f.endswith(".avi")])

        print(f"\n[{activity.upper()}] -- {len(video_files)} videos")

        for video_file in tqdm(video_files, desc=f"  {activity}", unit="video"):
            video_path  = os.path.join(activity_folder, video_file)
            output_file = os.path.join(output_folder, video_file.replace(".avi", ".npy"))

            # Skip if already processed (allows resuming)
            if os.path.exists(output_file):
                total_processed += 1
                continue

            keypoints = extract_video_keypoints(video_path, pose_model, api_version)

            if keypoints is not None and len(keypoints) > 0:
                np.save(output_file, keypoints)
                total_processed += 1
            else:
                total_skipped += 1

    # Clean up
    if api_version == "legacy":
        pose_model.close()
    else:
        pose_model.close()

    print("\n" + "=" * 60)
    print("  [DONE] Extraction complete!")
    print(f"  Processed : {total_processed} videos")
    print(f"  Skipped   : {total_skipped} videos")
    print(f"  Output    : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    process_all_videos()
