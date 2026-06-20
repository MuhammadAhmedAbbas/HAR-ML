# ⚠️ DEPRECATED — Please use the IMPROVED Version

**This original README is outdated. Please read the improved documentation:**

## 📖 Documentation (Read These First)

1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ — Get running in 5 minutes
2. **[README_IMPROVED.md](README_IMPROVED.md)** — Full technical documentation
3. **[MIGRATION_GUIDE.txt](MIGRATION_GUIDE.txt)** — What changed and why
4. **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** — Complete list of fixes

---

# 🏃 Human Activity Recognition (HAR) Desktop Application - IMPROVED VERSION
### Final Semester Project — Machine Learning Lab
**University Project | Python | OpenCV | MediaPipe | TensorFlow | PyQt5**

---

## ⚡ Quick Links

- 🚀 **Quick Start:** `python quick_train_improved.py` then `python gui/app_improved.py`
- 📊 **Full Setup:** `python setup_and_train.py`
- 📖 **Full Docs:** See [README_IMPROVED.md](README_IMPROVED.md)

---

## 📋 Overview

This application recognizes **6 human activities** from video files:

| Activity | Label |
|---|---|
| Walking | 0 |
| Running | 1 |
| Jogging | 2 |
| Boxing | 3 |
| Hand Waving | 4 |
| Hand Clapping | 5 |

**Pipeline:**
```
KTH Videos → MediaPipe Pose Keypoints → Sliding Windows → LSTM / Random Forest → GUI Prediction
```

---

## 📁 Project Structure

```
Human Activity Recognization/
├── KTH dataset/                  ← Raw KTH videos (already present)
│   ├── boxing/
│   ├── handclapping/
│   ├── handwaving/
│   ├── jogging/
│   ├── running/
│   └── walking/
├── dataset/                      ← Generated .npy keypoint files
│   ├── keypoints/                ← Per-video keypoint sequences
│   ├── X.npy                     ← Training sequences (N, 30, 132)
│   └── y.npy                     ← Labels (N,)
├── models/                       ← Saved trained models
│   ├── har_lstm_model.h5
│   ├── har_rf_model.pkl
│   └── label_encoder.pkl
├── preprocessing/
│   ├── extract_keypoints.py      ← Step 1: Extract MediaPipe keypoints
│   ├── normalize_features.py     ← Step 2: Normalize + window sequences
│   └── prepare_dataset.py        ← Master script (runs Step 1 + 2)
├── training/
│   ├── train_lstm.py             ← Train LSTM (Colab recommended)
│   ├── train_random_forest.py    ← Train RF (local, fast)
│   └── evaluate_model.py         ← Metrics + confusion matrix
├── gui/
│   └── app.py                    ← PyQt5 desktop application
├── utils/
│   ├── predictor.py              ← Inference engine
│   ├── logger.py                 ← Prediction logging
│   └── visualizer.py             ← Frame annotation
├── logs/                         ← Auto-created prediction logs
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Create and activate a virtual environment
```bash
python -m venv har_env
# Windows:
har_env\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Step-by-Step Workflow

### Step 1 — Preprocess the KTH Dataset
This extracts MediaPipe pose landmarks from all 600 videos and builds training sequences.

> ⏱️ Takes 15–30 minutes on CPU. Progress is saved incrementally.

```bash
python preprocessing/prepare_dataset.py
```

After this step, `dataset/X.npy` and `dataset/y.npy` will be ready.

---

### Step 2 — Train the Model

#### Option A: Train Random Forest (Local — ~5 min)
```bash
python training/train_random_forest.py
```
Saves `models/har_rf_model.pkl` and `models/label_encoder.pkl`.

#### Option B: Train LSTM (Google Colab — Recommended for best accuracy)
1. Upload your project to Google Drive or Colab
2. Upload `dataset/X.npy` and `dataset/y.npy`
3. Open `training/train_lstm.py` in a Colab notebook and run it
4. Download `models/har_lstm_model.h5` and `models/label_encoder.pkl`
5. Place them in the `models/` folder of this project

---

### Step 3 — Evaluate the Model
```bash
python training/evaluate_model.py
```
Outputs:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix saved as `models/confusion_matrix.png`

---

### Step 4 — Run the Desktop Application
```bash
python gui/app.py
```

---

## 🖥️ GUI Features

| Button | Action |
|---|---|
| 📁 Upload Video | Select a video file (.mp4, .avi) |
| 📷 Start Webcam | Start live activity recognition |
| ⏹ Stop Detection | Stop current feed |
| 🚪 Exit | Close the application |

- **Activity Display**: Large label showing detected activity
- **Confidence Bar**: Visual percentage confidence
- **Log Panel**: Real-time prediction history
- **Status Bar**: Error messages and system status

---

## ⚠️ Error Handling

| Scenario | Behavior |
|---|---|
| No person in frame | Shows "No Person Detected" |
| Invalid video file | Shows error dialog |
| Webcam unavailable | Shows error dialog |
| Model not found | Shows setup instructions dialog |

---

## 📊 Performance Metrics

Run `training/evaluate_model.py` to generate:
- **Accuracy** on held-out test set
- **Confusion Matrix** plot (saved as PNG)
- **Classification Report** (precision, recall, F1 per class)
- All metrics saved to `logs/evaluation_report.txt`

---

## 🔧 Technical Details

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Video Processing | OpenCV |
| Pose Estimation | MediaPipe Pose (33 landmarks × 4 = 132 features) |
| Sequence Length | 30 frames per window |
| Primary Model | LSTM (128 → 64 → Dense → 6) |
| Fallback Model | Random Forest (200 estimators) |
| GUI Framework | PyQt5 |
| Logs | `logs/predictions.log` |

---

## 📝 Authors

University Final Semester Project — Machine Learning Lab
