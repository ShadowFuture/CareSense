import json
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DB_PATH = "health_database.db"
WINDOW_NAME = "PulseONLINE Webcam"
MAX_SIGNAL_SECONDS = 12
PULSE_OUTPUT_FILE = Path(__file__).resolve().parent / "pulse.json"


# =====================================================================
# 1. DATABASE SETUP
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cardiovascular_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            estimated_bpm REAL,
            event_type TEXT,
            risk_score INTEGER,
            confidence REAL,
            warnings TEXT,
            recommendation TEXT,
            short_summary TEXT,
            long_summary TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO cardiovascular_predictions (
            timestamp, estimated_bpm, event_type, risk_score, confidence,
            warnings, recommendation, short_summary, long_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["timestamp"],
            result["estimated_bpm"],
            result["event_type"],
            result["risk_score"],
            result["confidence"],
            result["warnings"],
            result["recommendation"],
            result["short_summary"],
            result["long_summary"],
        ),
    )
    conn.commit()
    conn.close()


def write_pulse_output(result, image_quality, signal_samples, device_index):
    payload = {
        "source": "pulse_online",
        "timestamp": result["timestamp"],
        "prediction": result,
        "capture": {
            "device_index": device_index,
            "image_quality": image_quality,
            "signal_sample_count": len(signal_samples),
            "signal_samples": list(signal_samples),
        },
    }
    try:
        with open(PULSE_OUTPUT_FILE, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.flush()
    except OSError as e:
        print(f"Error writing to pulse file: {e}")


# =====================================================================
# 2. ML INFERENCE ENGINE
# =====================================================================
def calculate_dynamic_confidence(bpm, image_quality, signal_buffer, face_detected):
    if bpm is None:
        return 0.20

    s_quality = float(np.clip(image_quality, 0.5, 1.0))

    if signal_buffer is not None and len(signal_buffer) > 10:
        diffs = np.diff(signal_buffer)
        signal_noise = np.std(diffs)
        s_stability = float(np.clip(1.0 - (signal_noise / 12.0), 0.5, 1.0))
    else:
        s_stability = 0.6

    if 55 <= bpm <= 100:
        s_vital = 1.0
    elif 45 <= bpm < 55 or 100 < bpm <= 130:
        s_vital = 0.85
    else:
        s_vital = 0.70

    face_bonus = 0.25 if face_detected else 0.0
    raw_confidence = (s_quality * 0.3) + (s_stability * 0.3) + (s_vital * 0.15) + face_bonus
    return round(float(np.clip(raw_confidence, 0.3, 0.98)), 2)


def predict_cardiovascular_health(bpm, image_quality, signal_buffer, face_detected):
    confidence = calculate_dynamic_confidence(bpm, image_quality, signal_buffer, face_detected)

    if bpm is None:
        event_type = "normal"
        risk_score = 0
        warnings = "Accumulating signal for pulse calculation..."
        recommendation = "Keep your head still and look into the camera"
        short_summary = "Waiting for usable pulse signal."
        long_summary = "Signal buffering in progress."
    elif bpm > 130 or bpm < 45:
        event_type = "abnormal_movement"
        risk_score = 85
        warnings = "Severe tachycardia or bradycardia pattern detected"
        recommendation = "Seek clinical verification and rest"
        short_summary = "Emergency: Severe heart rate anomaly flagged."
        long_summary = "Extracted heart rate falls outside standard physiological limits."
    else:
        event_type = "normal"
        risk_score = 10
        warnings = "No critical heart-rate event detected"
        recommendation = "Continue monitoring"
        short_summary = "Normal heart-rate pattern detected."
        long_summary = "Estimated pulse is within expected normal range."

    return {
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
        "estimated_bpm": None if bpm is None else round(float(bpm), 1),
        "event_type": event_type,
        "risk_score": risk_score,
        "confidence": confidence,
        "warnings": warnings,
        "recommendation": recommendation,
        "short_summary": short_summary,
        "long_summary": long_summary,
    }


# =====================================================================
# 3. WEBCAM & SIGNAL PROCESSING
# =====================================================================
def open_camera(device_index=0):
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cam = cv2.VideoCapture(device_index, backend)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                return cam
            cam.release()
    raise RuntimeError(f"Unable to open camera at index {device_index}")


def _load_face_cascade():
    candidate_paths = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml",
    ]
    for cascade_path in candidate_paths:
        if cascade_path.exists():
            face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if not face_cascade.empty():
                return face_cascade
    return None


def _find_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _extract_pulse_sample(frame, face_box=None):
    """
    Extracts normalized Green ratio G / (R + G + B) to neutralize light changes.
    """
    height, width = frame.shape[:2]

    if face_box is not None:
        x, y, face_width, face_height = face_box
        roi_x1 = max(0, x + int(face_width * 0.2))
        roi_y1 = max(0, y + int(face_height * 0.05))
        roi_x2 = min(width, x + int(face_width * 0.8))
        roi_y2 = min(height, y + int(face_height * 0.35))
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    else:
        roi = frame[int(height * 0.2):int(height * 0.5), int(width * 0.3):int(width * 0.7)]

    if roi.size == 0:
        return None

    # Compute mean RGB values
    mean_b = np.mean(roi[:, :, 0])
    mean_g = np.mean(roi[:, :, 1])
    mean_r = np.mean(roi[:, :, 2])

    total = mean_r + mean_g + mean_b
    if total == 0:
        return None

    # Return normalized green signal ratio
    return float(mean_g / total)


def calculate_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    sharpness_score = float(np.clip(sharpness / 120.0, 0.4, 1.0))
    lighting_score = 0.6 if (brightness < 40 or brightness > 220) else 1.0
    return round(float(sharpness_score * lighting_score), 2)


def _estimate_bpm_from_signal(signal_samples, time_samples):
    """
    Filters out motion drift and calculates true BPM peak.
    """
    if len(signal_samples) < 60:  # Requires ~2 seconds of frames
        return None

    times = np.asarray(time_samples, dtype=float)
    values = np.asarray(signal_samples, dtype=float)
    times = times - times[0]
    duration = times[-1]

    if duration < 2.0:
        return None

    # Resample evenly
    sample_rate = len(values) / duration
    sample_rate = float(np.clip(sample_rate, 15.0, 30.0))
    uniform_times = np.linspace(0.0, duration, int(duration * sample_rate))
    uniform_values = np.interp(uniform_times, times, values)

    # DETRENDING: Subtract moving average to eliminate low-frequency drift
    window_size = int(sample_rate * 1.2)
    if window_size % 2 == 0:
        window_size += 1

    if len(uniform_values) > window_size:
        moving_avg = np.convolve(uniform_values, np.ones(window_size) / window_size, mode="same")
        detrended_values = uniform_values - moving_avg
    else:
        detrended_values = uniform_values - np.mean(uniform_values)

    # Apply Hanning Window
    window = np.hanning(len(detrended_values))
    spectrum = np.abs(np.fft.rfft(detrended_values * window))
    frequencies = np.fft.rfftfreq(len(detrended_values), d=1.0 / sample_rate)

    # Restrict frequency band between 0.95 Hz (57 BPM) and 2.5 Hz (150 BPM)
    band = (frequencies >= 0.95) & (frequencies <= 2.5)
    if not np.any(band):
        return None

    band_frequencies = frequencies[band]
    band_spectrum = spectrum[band]

    if len(band_spectrum) == 0:
        return None

    peak_frequency = float(band_frequencies[np.argmax(band_spectrum)])
    bpm = peak_frequency * 60.0

    return round(float(bpm), 1)


def capture_pulse_live(device_index=0):
    init_db()
    face_cascade = _load_face_cascade()
    camera = open_camera(device_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    signal_samples = deque(maxlen=int(MAX_SIGNAL_SECONDS * 30))
    time_samples = deque(maxlen=int(MAX_SIGNAL_SECONDS * 30))
    last_result = None

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError("Webcam opened, but no frame could be read")

            face_box = _find_face(frame, face_cascade) if face_cascade is not None else None
            face_detected = face_box is not None

            pulse_sample = _extract_pulse_sample(frame, face_box)
            if pulse_sample is not None:
                signal_samples.append(pulse_sample)
                time_samples.append(time.time())

            bpm = _estimate_bpm_from_signal(signal_samples, time_samples)
            image_quality = calculate_image_quality(frame)
            result = predict_cardiovascular_health(bpm, image_quality, list(signal_samples), face_detected)

            save_prediction(result)
            write_pulse_output(result, image_quality, list(signal_samples), device_index)
            last_result = result

            display_frame = frame.copy()
            if face_box is not None:
                x, y, w, h = face_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            overlay_lines = [
                f"BPM: {result['estimated_bpm'] if result['estimated_bpm'] is not None else 'Calculating...'}",
                f"Event: {result['event_type']}",
                f"Confidence: {result['confidence']}",
            ]

            y_position = 30
            for line in overlay_lines:
                cv2.putText(
                    display_frame,
                    line,
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y_position += 30

            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return last_result
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        result = capture_pulse_live()
        print("Pulse output:")
        print(result)
    except RuntimeError as error:
        print(f"Pulse webcam error: {error}")
