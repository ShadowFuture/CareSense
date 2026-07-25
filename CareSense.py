import json
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DB_PATH = "health_database.db"
WINDOW_NAME = "F.A.S.T. Stabilized Screen"
PULSE_OUTPUT_FILE = Path(__file__).resolve().parent / "pulse.json"

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
MOUTH_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stroke_fast_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            facial_droop_score REAL,
            arm_drift_score REAL,
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
        INSERT INTO stroke_fast_predictions (
            timestamp, facial_droop_score, arm_drift_score, event_type,
            risk_score, confidence, warnings, recommendation,
            short_summary, long_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["timestamp"],
            result["facial_droop_score"],
            result["arm_drift_score"],
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


def write_pulse_output(result, image_quality, device_index):
    payload = {
        "source": "fast_stroke_screening",
        "timestamp": result["timestamp"],
        "facial_droop_score": result["facial_droop_score"],
        "arm_drift_score": result["arm_drift_score"],
        "prediction": result,
        "capture": {
            "device_index": device_index,
            "image_quality": image_quality,
        },
    }
    try:
        with open(PULSE_OUTPUT_FILE, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.flush()
    except OSError as e:
        print(f"Error writing output file: {e}")


def calculate_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    sharpness_score = float(np.clip(sharpness / 150.0, 0.2, 1.0))
    lighting_score = 0.5 if (brightness < 40 or brightness > 220) else 1.0
    return round(float(sharpness_score * lighting_score), 2)


def evaluate_facial_droop_normalized(face_gray, face_w, face_h):
    upper_face = face_gray[0:int(face_h * 0.55), :]
    eyes = EYE_CASCADE.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))

    lower_face = face_gray[int(face_h * 0.50):face_h, :]
    mouths = MOUTH_CASCADE.detectMultiScale(lower_face, scaleFactor=1.4, minNeighbors=12, minSize=(30, 20))

    if len(eyes) < 2:
        return None

    eyes = sorted(eyes, key=lambda e: e[0])
    e1_x, e1_y = eyes[0][0] + eyes[0][2] / 2, eyes[0][1] + eyes[0][3] / 2
    e2_x, e2_y = eyes[1][0] + eyes[1][2] / 2, eyes[1][1] + eyes[1][3] / 2

    delta_x = e2_x - e1_x
    delta_y = e2_y - e1_y
    if delta_x == 0:
        return None
    tilt_angle = np.arctan2(delta_y, delta_x)

    corrected_eye_disparity = abs(delta_y - (delta_x * np.tan(tilt_angle)))
    droop_score = corrected_eye_disparity / face_h

    if len(mouths) > 0:
        mouth = mouths[0]
        mouth_center_y = mouth[1] + mouth[3] / 2
        expected_y = face_h * 0.25
        droop_score += abs(mouth_center_y - expected_y) / face_h

    return round(float(droop_score), 3)


def predict_fast_risk(droop_score, arm_drift_score, baseline_droop, image_quality):
    net_droop = max(0.0, droop_score - baseline_droop)
    base_confidence = float(np.clip(image_quality, 0.3, 0.95))

    if net_droop > 0.08 and arm_drift_score > 0.12:
        event_type = "high_risk_fast_flagged"
        risk_score = min(95, int(50 + (net_droop * 300) + (arm_drift_score * 150)))
        warnings = "Sustained facial droop and lower body drift detected"
        recommendation = "Seek clinical evaluation or emergency medical attention"
        short_summary = "CRITICAL: Significant asymmetry and movement drift."
        long_summary = "Multiple physical markers show significant deviation from your baseline."
    elif net_droop > 0.06:
        event_type = "facial_droop_detected"
        risk_score = min(75, int(30 + (net_droop * 400)))
        warnings = "Noticeable facial asymmetry detected compared to baseline"
        recommendation = "Verify facial alignment and remain seated"
        short_summary = "WARNING: Facial droop threshold exceeded."
        long_summary = "Facial alignment deviates significantly from baseline calibration."
    elif arm_drift_score > 0.12:
        event_type = "body_drift_detected"
        risk_score = min(50, int(20 + (arm_drift_score * 200)))
        warnings = "Unsteady lower body motion detected"
        recommendation = "Maintain steady positioning"
        short_summary = "Noticeable motion drift."
        long_summary = "High frame variance detected in torso region."
    else:
        event_type = "normal"
        risk_score = max(5, int(10 + (net_droop * 50)))
        warnings = "No critical stroke indicators detected"
        recommendation = "Continue monitoring"
        short_summary = "Normal facial alignment and motion."
        long_summary = "Readings remain aligned with your calibrated baseline."

    return {
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
        "facial_droop_score": round(net_droop, 3),
        "arm_drift_score": round(arm_drift_score, 3),
        "event_type": event_type,
        "risk_score": risk_score,
        "confidence": base_confidence,
        "warnings": warnings,
        "recommendation": recommendation,
        "short_summary": short_summary,
        "long_summary": long_summary,
    }


def open_camera(device_index=0):
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cam = cv2.VideoCapture(device_index, backend)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                return cam
            cam.release()

    raise RuntimeError(
        f"Unable to open camera at index {device_index}. Ensure no other application is using the webcam."
    )


def run_fast_stabilized_loop(device_index=0):
    init_db()
    camera = open_camera(device_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    droop_buffer = deque(maxlen=15)
    drift_buffer = deque(maxlen=15)

    print("Calibrating facial baseline... Look directly at the camera.")
    calibration_samples = []
    calib_start = time.time()

    while time.time() - calib_start < 3.0:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            score = evaluate_facial_droop_normalized(gray[y:y + h, x:x + w], w, h)
            if score is not None:
                calibration_samples.append(score)

        cv2.putText(frame, "CALIBRATING... STAY STILL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    baseline_droop = float(np.mean(calibration_samples)) if calibration_samples else 0.02
    print(f"Calibration Complete. Personal Baseline Score: {baseline_droop:.3f}")

    prev_gray = None
    last_result = None

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                face_box = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)

                raw_droop = evaluate_facial_droop_normalized(gray[y:y + h, x:x + w], w, h)
                if raw_droop is not None:
                    droop_buffer.append(raw_droop)

                if prev_gray is not None:
                    fx, fy, fw, fh = face_box
                    b_y1, b_y2 = min(frame.shape[0], fy + fh), min(frame.shape[0], fy + int(fh * 2.5))
                    b_x1, b_x2 = max(0, fx - int(fw * 0.5)), min(frame.shape[1], fx + int(fw * 1.5))
                    if b_y2 > b_y1 and b_x2 > b_x1:
                        diff = cv2.absdiff(gray[b_y1:b_y2, b_x1:b_x2], prev_gray[b_y1:b_y2, b_x1:b_x2])
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        drift_buffer.append(np.sum(thresh) / (thresh.size * 255.0))

            prev_gray = gray.copy()

            avg_droop = float(np.mean(droop_buffer)) if droop_buffer else baseline_droop
            avg_drift = float(np.mean(drift_buffer)) if drift_buffer else 0.0

            image_quality = calculate_image_quality(frame)
            result = predict_fast_risk(avg_droop, avg_drift, baseline_droop, image_quality)

            save_prediction(result)
            write_pulse_output(result, image_quality, device_index)
            last_result = result

            color = (0, 0, 255) if result["risk_score"] > 40 else (0, 255, 0)

            # Updated overlay array (Net Droop line removed)
            overlay = [
                f"Status: {result['event_type']}",
                f"Dynamic Risk Score: {result['risk_score']}",
                f"Motion Drift: {result['arm_drift_score']}",
            ]
            
            y_pos = 30
            for line in overlay:
                cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 26

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return last_result
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_fast_stabilized_loop()
    except Exception as err:
        print(f"Execution Error: {err}")
    
