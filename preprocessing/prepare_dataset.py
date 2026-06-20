"""
preprocessing/prepare_dataset.py
==================================
MASTER preprocessing script.

PURPOSE:
    Orchestrates the entire preprocessing pipeline in one command:
      1. Extract MediaPipe pose keypoints from all KTH videos
      2. Normalize features and build windowed training sequences

USAGE:
    py -3 preprocessing/prepare_dataset.py

    Options:
      --skip-extraction     Skip Step 1 if keypoints already extracted
      --skip-normalization  Skip Step 2 if X.npy/y.npy already exist
"""

import os
import sys
import argparse
import time

# Fix Unicode output on Windows (prevents UnicodeEncodeError)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path so we can import from sibling packages
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_keypoints import process_all_videos
from preprocessing.normalize_features import run_normalization


def check_prerequisites():
    """
    Verify that the KTH dataset folder exists before starting.
    Prints a clear error message if not found.
    """
    dataset_path = os.path.join(PROJECT_ROOT, "KTH dataset")

    if not os.path.exists(dataset_path):
        print("=" * 60)
        print("  [ERROR] KTH dataset folder not found!")
        print(f"  Expected at: {dataset_path}")
        print()
        print("  Please ensure your folder structure is:")
        print("  Human Activity Recognization/")
        print("  └── KTH dataset/")
        print("      ├── boxing/")
        print("      ├── handclapping/")
        print("      ├── handwaving/")
        print("      ├── jogging/")
        print("      ├── running/")
        print("      └── walking/")
        print("=" * 60)
        sys.exit(1)

    print(f"  [OK] KTH dataset found at: {dataset_path}")
    return True


def print_banner():
    """Print a formatted banner at startup."""
    print()
    print("=" * 60)
    print("  Human Activity Recognition -- Dataset Preparation")
    print("  KTH Action Dataset -> MediaPipe Keypoints -> .npy")
    print("=" * 60)
    print()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess KTH dataset for HAR training"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip keypoint extraction (use if dataset/keypoints/ already exists)"
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalization (use if dataset/X.npy already exists)"
    )
    args = parser.parse_args()

    print_banner()

    # Check prerequisites
    check_prerequisites()

    total_start = time.time()

    # Step 1: Extract keypoints
    if not args.skip_extraction:
        keypoints_path = os.path.join(PROJECT_ROOT, "dataset", "keypoints")
        if os.path.exists(keypoints_path) and len(os.listdir(keypoints_path)) > 0:
            print("\n  [INFO] Keypoints folder already exists.")
            print("  Tip: Use --skip-extraction to skip this step next time.")

        print("\n" + "-" * 60)
        print("  STEP 1/2: Extracting MediaPipe Pose Keypoints")
        print("  (This takes 15-30 minutes on CPU. Progress is saved.)")
        print("-" * 60)

        step1_start = time.time()
        process_all_videos()
        step1_time = time.time() - step1_start
        print(f"\n  Step 1 completed in {step1_time/60:.1f} minutes")

    else:
        print("\n  [SKIPPED] Keypoint extraction (--skip-extraction flag set)")

    # Step 2: Normalize and build dataset
    if not args.skip_normalization:
        print("\n" + "-" * 60)
        print("  STEP 2/2: Normalizing Features & Building Training Dataset")
        print("-" * 60)

        step2_start = time.time()
        run_normalization()
        step2_time = time.time() - step2_start
        print(f"\n  Step 2 completed in {step2_time:.1f} seconds")

    else:
        print("\n  [SKIPPED] Normalization (--skip-normalization flag set)")

    # Summary
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print("  [DONE] Preprocessing Pipeline Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print()
    print("  Next Steps:")
    print("  1. Train RF locally (~5-10 min):")
    print("     py -3 training/train_random_forest.py")
    print("  2. Or train LSTM on Google Colab:")
    print("     Upload dataset/X.npy + y.npy, run train_lstm.py")
    print("  3. Run the GUI app:")
    print("     py -3 gui/app.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
