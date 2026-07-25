import sqlite3
from datetime import datetime
import cv2
import numpy as np

# =====================================================================
# 1. DATABASE SETUP
# =====================================================================
def init_db():
    conn = sqlite3.connect("health_database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cardiovascular_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            estimated_bpm REAL,
            estimated_hrv REAL,
            event_type TEXT,
            risk_score INTEGER,
            confidence REAL,
            warnings TEXT,
            recommendation TEXT,
            short_summary TEXT,
            long_summary TEXT
        )
    """)
    conn.commit()
    conn.close()

# =====================================================================
# 2. YOUR ML INFERENCE ENGINE
# =====================================================================
def calculate_dynamic_confidence(bpm, hrv, image_quality, signal_buffer):
    """
    Calculates a dynamic confidence score (0.0 to 1.0) 
    based on raw signal stability, vital plausibility, and video sharpness.
    """
    if bpm is None:
        return 0.0

    # 1. Image Quality Score (Sharpness / Lighting)
    s_quality = float(np.clip(image_quality, 0.2, 1.0))

    # 2. Signal Noise / Stability Score
    if signal_buffer is not None and len(signal_buffer) > 10:
        diffs = np.diff(signal_buffer)
        signal_noise = np.std(diffs)
        s_stability = float(np.clip(1.0 - (signal_noise / 5.0), 0.1, 1.0))
    else:
        s_stability = 0.5

    # 3. Physiological Plausibility Score
    if 55 <= bpm <= 100:
        s_vital = 1.0
    elif 45 <= bpm < 55 or 100 < bpm <= 130:
        s_vital = 0.85
    else:
        s_vital = 0.60

    # Calculate overall varied confidence
    varied_confidence = (s_quality * 0.4) + (s_stability * 0.4) + (s_vital * 0.2)
    return round(float(varied_confidence), 2)


def predict_cardiovascular_health(bpm, hrv, image_quality, signal_buffer):
    """
    ML Cardiovascular Engine with fully dynamic confidence scoring.
    """
    confidence = calculate_dynamic_confidence(bpm, hrv, image_quality, signal_buffer)

    if bpm > 130 or bpm < 45:
        event_type = "abnormal_movement"
        risk_score = 85
        warnings = "Severe tachycardia or bradycardia pattern detected"
        recommendation = "Seek immediate clinical verification and rest"
        short_summary = "Emergency: Severe heart rate anomaly flagged."
