"""
training/train_random_forest.py
=================================
Train a Random Forest classifier for Human Activity Recognition.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
X_path = os.path.join(MODELS_PATH, "X_all.npy")
y_path = os.path.join(MODELS_PATH, "y_all.npy")

if not os.path.exists(X_path):
    print("Data not found. Please run preprocessing first.")
    exit(1)

X = np.load(X_path)
y = np.load(y_path)

# Extract features
X_mean = X.mean(axis=1)
X_std = X.std(axis=1)
X_min = X.min(axis=1)
X_max = X.max(axis=1)
X_features = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)

print(f"Training Random Forest on {X_features.shape[0]} sequences...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_features, y)

preds = model.predict(X_features)
acc = accuracy_score(y, preds)
print(f"Training Accuracy: {acc:.4f}")

model_out = os.path.join(MODELS_PATH, "har_rf_model.pkl")
joblib.dump(model, model_out)
print(f"Model saved to {model_out}")