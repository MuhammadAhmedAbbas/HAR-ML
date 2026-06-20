"""
preprocessing/normalize_features.py
=====================================
Step 2 of the HAR preprocessing pipeline.

PURPOSE:
    Apply Z-score normalization using StandardScaler and sliding windows.
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

def normalize_all():
    print("Normalization script placeholder. In the real pipeline this builds X_all and y_all.")

if __name__ == "__main__":
    normalize_all()