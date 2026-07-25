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
MAX_SIGNAL_SECONDS = 15
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
            estimated_hrv REAL,
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
            timestamp,
            estimated_bpm,
            estimated_hrv,
            event_type,
            risk_score,
            confidence,
            warnings,
            recommendation,
            short_summary,
            long_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["timestamp"],
            result["estimated_bpm"],
            result["estimated_hrv"],
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

    with open(PULSE_OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.flush()


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

    s_quality = float(np.clip(image_quality, 0.2, 1.0))

    if signal_buffer is not None and len(signal_buffer) > 10:
        diffs = np.diff(signal_buffer)
        signal_noise = np.std(diffs)
        s_stability = float(np.clip(1.0 - (signal_noise / 5.0), 0.1, 1.0))
    else:
        s_stability = 0.5

    if 55 <= bpm <= 100:
        s_vital = 1.0
    elif 45 <= bpm < 55 or 100 < bpm <= 130:
        s_vital = 0.85
    else:
        s_vital = 0.60

    varied_confidence = (s_quality * 0.4) + (s_stability * 0.4) + (s_vital * 0.2)
    return round(float(varied_confidence), 2)


def predict_cardiovascular_health(bpm, hrv, image_quality, signal_buffer):
    """
    ML Cardiovascular Engine with fully dynamic confidence scoring.
    """
    confidence = calculate_dynamic_confidence(bpm, hrv, image_quality, signal_buffer)

    if bpm is None:
        event_type = "normal"
        risk_score = 0
        warnings = "No heart-rate signal detected yet"
        recommendation = "Adjust the webcam and keep the face visible"
        short_summary = "Waiting for a usable pulse signal."
        long_summary = (
            "The camera is active, but a stable pulse signal has not been extracted yet. "
            "Keep the face in view and stay steady for a few seconds."
        )
    elif bpm > 130 or bpm < 45:
        event_type = "abnormal_movement"
        risk_score = 85
        warnings = "Severe tachycardia or bradycardia pattern detected"
        recommendation = "Seek immediate clinical verification and rest"
        short_summary = "Emergency: Severe heart rate anomaly flagged."
        long_summary = (
            "The webcam signal suggests a heart-rate pattern outside the normal range. "
            "This result should be verified by a clinical device or professional assessment."
        )
    else:
        event_type = "normal"
        risk_score = 10
        warnings = "No critical heart-rate event detected"
        recommendation = "Continue monitoring"
        short_summary = "Normal heart-rate pattern detected."
        long_summary = (
            "The webcam-based estimate is within the expected range and does not indicate "
            "a critical cardiovascular event."
        )

    return {
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
        "estimated_bpm": None if bpm is None else round(float(bpm), 1),
        "estimated_hrv": None if hrv is None else round(float(hrv), 1),
        "event_type": event_type,
        "risk_score": risk_score,
        "confidence": confidence,
        "warnings": warnings,
        "recommendation": recommendation,
        "short_summary": short_summary,
        "long_summary": long_summary,
    }


# =====================================================================
# 3. WEBCAM PULSE EXTRACTION
# =====================================================================
def _open_camera(device_index=0):
    camera = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(device_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open webcam at index {device_index}")
    return camera


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

    print("Pulse webcam error: Face cascade failed to load")
    return None


def _find_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _extract_pulse_sample(frame, face_box=None):
    height, width = frame.shape[:2]

    if face_box is not None:
        x, y, face_width, face_height = face_box
        roi_x1 = max(0, x)
        roi_y1 = max(0, y)
        roi_x2 = min(width, x + face_width)
        roi_y2 = min(height, y + face_height)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    else:
        roi = frame[int(height * 0.2):int(height * 0.75), int(width * 0.25):int(width * 0.75)]

    if roi.size == 0:
        return None

    h = roi.shape[0]
    forehead = roi[: max(1, h // 3), :]
    if forehead.size == 0:
        forehead = roi

    return float(np.mean(forehead[:, :, 1]))


def calculate_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    sharpness_score = float(np.clip(sharpness / 150.0, 0.2, 1.0))
    if brightness < 40 or brightness > 220:
        lighting_score = 0.5
    else:
        lighting_score = 1.0

    return round(float(sharpness_score * lighting_score), 2)


def _estimate_hr_from_signal(signal_samples, time_samples):
    if len(signal_samples) < 30:
        return None, None

    times = np.asarray(time_samples, dtype=float)
    values = np.asarray(signal_samples, dtype=float)
    times = times - times[0]
    duration = times[-1]
    if duration <= 0:
        return None, None

    sample_rate = len(values) / duration
    sample_rate = float(np.clip(sample_rate, 15.0, 30.0))
    uniform_times = np.linspace(0.0, duration, max(2, int(duration * sample_rate)))
    uniform_values = np.interp(uniform_times, times, values)
    uniform_values = uniform_values - np.mean(uniform_values)

    if np.std(uniform_values) < 1e-6:
        return None, None

    window = np.hanning(len(uniform_values))
    spectrum = np.abs(np.fft.rfft(uniform_values * window))
    frequencies = np.fft.rfftfreq(len(uniform_values), d=1.0 / sample_rate)

    band = (frequencies >= 0.8) & (frequencies <= 3.0)
    if not np.any(band):
        return None, None

    band_frequencies = frequencies[band]
    band_spectrum = spectrum[band]
    peak_frequency = float(band_frequencies[np.argmax(band_spectrum)])
    bpm = peak_frequency * 60.0

    peaks = []
    threshold = float(np.mean(uniform_values) + 0.15 * np.std(uniform_values))
    for index in range(1, len(uniform_values) - 1):
        if (
            uniform_values[index] > uniform_values[index - 1]
            and uniform_values[index] > uniform_values[index + 1]
            and uniform_values[index] > threshold
        ):
            peaks.append(uniform_times[index])

    if len(peaks) >= 3:
        intervals = np.diff(peaks)
        hrv = float(np.std(intervals) * 1000.0)
    else:
        hrv = 0.0

    return round(float(bpm), 1), round(float(hrv), 1)


def capture_pulse_live(device_index=0):
    init_db()
    face_cascade = _load_face_cascade()
    camera = _open_camera(device_index)
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
            pulse_sample = _extract_pulse_sample(frame, face_box)
            if pulse_sample is not None:
                signal_samples.append(pulse_sample)
                time_samples.append(time.time())

            bpm, hrv = _estimate_hr_from_signal(signal_samples, time_samples)
            image_quality = calculate_image_quality(frame)
            result = predict_cardiovascular_health(bpm, hrv, image_quality, list(signal_samples))
            save_prediction(result)
            write_pulse_output(result, image_quality, list(signal_samples), device_index)
            last_result = result

            display_frame = frame.copy()
            overlay_lines = [
                f"BPM: {result['estimated_bpm'] if result['estimated_bpm'] is not None else '...'}",
                f"HRV: {result['estimated_hrv'] if result['estimated_hrv'] is not None else '...'}",
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

