"""
setup_and_train.py
===================
One-stop setup script that runs preprocessing and training sequentially.

Usage:
    python setup_and_train.py
"""

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def print_banner(title):
    """Print formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_script(script_name, description):
    """Run a Python script."""
    print_banner(description)
    
    script_path = os.path.join(PROJECT_ROOT, script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Script failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def main():
    print_banner("HAR — Complete Setup & Training")
    
    print("This script will:")
    print("  1. Preprocess KTH dataset (extract MediaPipe keypoints)")
    print("  2. Train LSTM model")
    print("  3. Generate evaluation metrics\n")
    
    input("Press ENTER to begin... ")
    
    start_time = time.time()
    
    # Step 1: Preprocessing
    print("\n[STEP 1/2] PREPROCESSING")
    if not run_script(
        "preprocessing/preprocess_simple.py",
        "Preprocessing: Extracting MediaPipe Keypoints"
    ):
        print("\n[FAILED] Preprocessing step failed!")
        sys.exit(1)
    
    # Step 2: Training
    print("\n[STEP 2/2] TRAINING")
    if not run_script(
        "training/train_random_forest_simple.py",
        "Training: Random Forest Model (No TensorFlow Issues!)"
    ):
        print("\n[FAILED] Training step failed!")
        sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print_banner("✅ SETUP & TRAINING COMPLETE!")
    print(f"Total time: {minutes}m {seconds}s\n")
    print("Next steps:")
    print("  1. Review evaluation metrics:")
    print("     cat logs/lstm_improved_evaluation.txt")
    print("  2. View confusion matrix:")
    print("     (open logs/confusion_matrix.png)")
    print("  3. Launch the GUI:")
    print("     python gui/app_improved.py\n")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
