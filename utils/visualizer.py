"""
utils/visualizer.py
=====================
Frame annotation utilities for the HAR application.

PURPOSE:
    Draw activity labels, confidence bars, frame info, and status
    overlays onto OpenCV frames before displaying in the GUI.

USAGE:
    from utils.visualizer import draw_activity_overlay, draw_no_person_message
    annotated = draw_activity_overlay(frame, label="walking", confidence=0.87)
"""

import cv2
import numpy as np

# ─── Color palette (BGR format for OpenCV) ─────────────────────────────────────

COLORS = {
    "boxing"        : (0,   69,  255),   # Red-orange
    "handclapping"  : (0,  215,  255),   # Gold
    "handwaving"    : (0,  255,  144),   # Spring green
    "jogging"       : (255, 144,   0),   # Sky blue
    "running"       : (255,   0,  128),  # Hot pink
    "walking"       : (144, 238, 144),   # Light green
    "default"       : (180, 180, 180),   # Gray for uncertain/initializing
    "error"         : (0,     0, 200),   # Red for errors
    "background"    : (20,   20,  20),   # Near-black for overlays
}

# Activity emoji icons (using text since OpenCV doesn't support emoji fonts)
ACTIVITY_ICONS = {
    "boxing"        : "[BOX]",
    "handclapping"  : "[CLP]",
    "handwaving"    : "[WAV]",
    "jogging"       : "[JOG]",
    "running"       : "[RUN]",
    "walking"       : "[WLK]",
}


def get_activity_color(label):
    """Return the BGR color for a given activity label."""
    label_lower = label.lower()
    for key, color in COLORS.items():
        if key in label_lower:
            return color
    return COLORS["default"]


def draw_activity_overlay(frame, label, confidence, model_type=None, frame_num=None):
    """
    Draw the full activity recognition overlay on a video frame.

    Draws:
    - Semi-transparent top banner with activity label
    - Confidence progress bar
    - Model type indicator (bottom right)
    - Frame counter (bottom left)

    Parameters
    ----------
    frame      : np.ndarray — BGR frame from OpenCV
    label      : str        — Activity label (e.g., "walking")
    confidence : float      — Confidence in [0, 1]
    model_type : str        — "lstm" or "rf" (optional)
    frame_num  : int        — Current frame number (optional)

    Returns
    -------
    np.ndarray — Annotated frame
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top banner background ──────────────────────────────────────────
    banner_h = 90
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLORS["background"], -1)

    # Blend overlay with frame for semi-transparency
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Activity label ─────────────────────────────────────────────────
    activity_color = get_activity_color(label)
    display_label  = label.upper()

    # Large bold activity name
    font_scale = 1.3 if len(display_label) <= 10 else 1.0
    cv2.putText(
        frame,
        display_label,
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        activity_color,
        2,
        cv2.LINE_AA
    )

    # ── Confidence bar ─────────────────────────────────────────────────
    bar_x     = 20
    bar_y     = 65
    bar_w     = min(w - 40, 300)
    bar_h     = 14
    bar_fill  = int(bar_w * confidence)

    # Background bar (dark gray)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # Filled portion (activity color)
    if bar_fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), activity_color, -1)

    # Bar border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    # Confidence percentage text
    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(
        frame,
        conf_text,
        (bar_x + bar_w + 8, bar_y + 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # ── Model type indicator (bottom right) ────────────────────────────
    if model_type:
        model_text  = f"Model: {'LSTM' if model_type == 'lstm' else 'Random Forest'}"
        text_size   = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x      = w - text_size[0] - 10
        text_y      = h - 10

        cv2.putText(
            frame,
            model_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )

    # ── Frame counter (bottom left) ────────────────────────────────────
    if frame_num is not None:
        cv2.putText(
            frame,
            f"Frame #{frame_num}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 120, 120),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_no_person_message(frame):
    """
    Draw a "No Person Detected" warning on the frame.

    Parameters
    ----------
    frame : np.ndarray — BGR frame

    Returns
    -------
    np.ndarray — Frame with warning overlay
    """
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), COLORS["background"], -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Warning text
    cv2.putText(
        frame,
        "NO PERSON DETECTED",
        (20, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        COLORS["error"],
        2,
        cv2.LINE_AA
    )

    # Sub-text
    cv2.putText(
        frame,
        "Move into camera frame",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    return frame


def draw_buffering_message(frame, current, total):
    """
    Draw a "Buffering" message while the frame buffer fills up.

    Parameters
    ----------
    frame   : np.ndarray
    current : int — current buffer length
    total   : int — required buffer length (SEQUENCE_LENGTH)
    """
    h, w = frame.shape[:2]

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), COLORS["background"], -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Text
    cv2.putText(
        frame,
        f"Buffering frames: {current}/{total}",
        (20, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (0, 215, 255),
        2,
        cv2.LINE_AA
    )

    # Progress bar
    bar_w   = min(w - 40, 280)
    fill    = int(bar_w * current / total)
    cv2.rectangle(frame, (20, 60), (20 + bar_w, 74), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(frame, (20, 60), (20 + fill, 74), (0, 215, 255), -1)
    cv2.rectangle(frame, (20, 60), (20 + bar_w, 74), (180, 180, 180), 1)

    return frame


def draw_error_message(frame, message):
    """
    Draw a red error message centered on the frame.

    Parameters
    ----------
    frame   : np.ndarray
    message : str
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark overlay on entire frame
    overlay = np.zeros_like(frame)
    overlay[:] = (20, 20, 20)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Error text centered
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    text_x    = (w - text_size[0]) // 2
    text_y    = h // 2

    cv2.putText(
        frame,
        "⚠ " + message,
        (text_x - 20, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        COLORS["error"],
        2,
        cv2.LINE_AA
    )

    return frame


def resize_frame_for_display(frame, target_width=640, target_height=480):
    """
    Resize a frame to fit in the GUI display area while maintaining aspect ratio.

    Parameters
    ----------
    frame         : np.ndarray — Input BGR frame
    target_width  : int
    target_height : int

    Returns
    -------
    np.ndarray — Resized frame, padded with black bars if needed
    """
    h, w = frame.shape[:2]

    # Calculate scaling factor
    scale_w = target_width  / w
    scale_h = target_height / h
    scale   = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create black canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center the resized frame on the canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width  - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas
