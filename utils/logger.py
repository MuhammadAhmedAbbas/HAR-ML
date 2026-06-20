"""
utils/logger.py
=================
PredictionLogger — saves activity predictions to a timestamped log file.

PURPOSE:
    Record all predictions made by the GUI application for:
    - Debugging and analysis
    - Creating a history of recognized activities
    - Academic reporting

LOG FILE: logs/predictions.log
FORMAT:   [2025-05-15 14:32:10] | ACTIVITY: walking       | CONFIDENCE: 87.3% | SOURCE: webcam

USAGE:
    logger = PredictionLogger()
    logger.log(label="walking", confidence=0.873, source="webcam")
    logger.close()
"""

import os
import sys
import logging
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH    = os.path.join(PROJECT_ROOT, "logs")

# Log file path
LOG_FILE = os.path.join(LOGS_PATH, "predictions.log")

# Minimum confidence to log (skip very uncertain predictions)
MIN_CONFIDENCE_TO_LOG = 0.30

# Maximum entries in memory (for the in-GUI log panel display)
MAX_IN_MEMORY_LOGS = 200


class PredictionLogger:
    """
    Manages prediction logging to a file and in-memory for GUI display.

    Features:
    - Thread-safe file logging using Python's logging module
    - In-memory ring buffer for displaying recent logs in the GUI
    - Session summary on close (total predictions, per-class counts)
    - Human-readable formatted log entries
    """

    def __init__(self):
        """Initialize logger: create log directory and set up handlers."""
        # Create logs directory if it doesn't exist
        os.makedirs(LOGS_PATH, exist_ok=True)

        # ── Set up Python logger ───────────────────────────────────────
        self.logger = logging.getLogger("HAR_Predictor")
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if logger already configured
        if not self.logger.handlers:
            # File handler — writes to predictions.log
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(message)s")  # We format messages ourselves
            )
            self.logger.addHandler(file_handler)

        # ── Session tracking ───────────────────────────────────────────
        self.session_start    = datetime.now()
        self.total_predictions = 0
        self.class_counts     = {}   # {label: count}
        self.recent_logs      = []   # In-memory for GUI display

        # Write session start marker to log file
        self._write_session_header()

    def _write_session_header(self):
        """Write a session start marker to the log file."""
        header = (
            "\n" + "=" * 70 + "\n"
            f"  SESSION STARTED: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "=" * 70
        )
        self.logger.info(header)

    def log(self, label, confidence, source="unknown"):
        """
        Log a single prediction.

        Parameters
        ----------
        label      : str   — Activity name (e.g., "walking")
        confidence : float — Confidence score [0, 1]
        source     : str   — "webcam" or filename of uploaded video
        """
        # Skip non-activity messages
        if label in ("No Person Detected", "Uncertain", "Initializing...") or \
           label.startswith("Buffering"):
            return

        # Skip very low confidence predictions
        if confidence < MIN_CONFIDENCE_TO_LOG:
            return

        self.total_predictions += 1
        self.class_counts[label] = self.class_counts.get(label, 0) + 1

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conf_pct  = confidence * 100

        # Build log line
        log_line = (
            f"[{timestamp}] | "
            f"ACTIVITY: {label:<15} | "
            f"CONFIDENCE: {conf_pct:>5.1f}% | "
            f"SOURCE: {source}"
        )

        # Write to file
        self.logger.info(log_line)

        # Keep in memory for GUI (ring buffer)
        self.recent_logs.append({
            "timestamp"  : timestamp,
            "label"      : label,
            "confidence" : conf_pct,
            "source"     : source,
            "display"    : f"{timestamp[-8:]}  {label:<15}  {conf_pct:.0f}%"
        })

        # Trim in-memory log to max size
        if len(self.recent_logs) > MAX_IN_MEMORY_LOGS:
            self.recent_logs.pop(0)

    def log_error(self, message):
        """Log an error/warning message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_line = f"[{timestamp}] | ERROR: {message}"
        self.logger.warning(error_line)

        self.recent_logs.append({
            "timestamp"  : timestamp,
            "label"      : "ERROR",
            "confidence" : 0,
            "source"     : "",
            "display"    : f"{timestamp[-8:]}  ⚠ {message[:40]}"
        })

    def get_recent_logs(self, n=50):
        """
        Return the N most recent log entries for display in the GUI.

        Returns
        -------
        list of str — formatted log lines for display
        """
        recent = self.recent_logs[-n:]
        return [entry["display"] for entry in reversed(recent)]

    def get_session_summary(self):
        """
        Return a formatted summary of the current session.

        Returns
        -------
        str — multi-line summary
        """
        duration = datetime.now() - self.session_start
        minutes  = int(duration.total_seconds() // 60)
        seconds  = int(duration.total_seconds() % 60)

        lines = [
            "─" * 40,
            f"Session Duration : {minutes}m {seconds}s",
            f"Total Predictions: {self.total_predictions}",
            "─" * 40,
        ]

        if self.class_counts:
            lines.append("Activity Counts:")
            for label, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
                bar_len = int(count / max(self.class_counts.values()) * 20)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {label:<15} {bar} {count}")

        return "\n".join(lines)

    def close(self):
        """Write session summary and close log file."""
        duration = datetime.now() - self.session_start
        minutes  = int(duration.total_seconds() // 60)
        seconds  = int(duration.total_seconds() % 60)

        footer = (
            "\n" + "─" * 70 + "\n"
            f"  SESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Duration          : {minutes}m {seconds}s\n"
            f"  Total Predictions : {self.total_predictions}\n"
        )

        if self.class_counts:
            footer += "  Activity Breakdown:\n"
            for label, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
                pct = count / self.total_predictions * 100 if self.total_predictions > 0 else 0
                footer += f"    {label:<15}: {count:>4}  ({pct:.1f}%)\n"

        footer += "=" * 70 + "\n"
        self.logger.info(footer)

        # Flush and close handlers
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        self.logger.handlers.clear()
