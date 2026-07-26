import json
from pathlib import Path

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None

engine = pyttsx3.init() if pyttsx3 is not None else None
BASE_DIR = Path(__file__).resolve().parent


def speak(text):
    if engine is None:
        return None
    engine.say(text)
    engine.runAndWait()
    return None

def read_json_file(filename):
    file_path = BASE_DIR / filename
    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return {}

    return json.loads(content)


# Read ML summary
def read_ml_summary():
    return read_json_file("ml_output.json")


# Read hardware JSON
def read_hw_data():
    return read_json_file("hardware_output.json")


def read_pulse_data():
    return read_json_file("pulse.json")


def build_summary():
    ml = read_ml_summary()
    hw = read_hw_data()
    rp = read_pulse_data()

    pulse_prediction = rp.get("prediction", rp)
    if pulse_prediction.get("estimated_bpm") is not None:
        pulse_line = (
            f"Pulse online reports {pulse_prediction['estimated_bpm']} beats per minute "
            f"with {pulse_prediction.get('estimated_hrv', 'unknown')} HRV."
        )
    else:
        pulse_line = "Pulse online is still waiting for a stable signal."

    summary = (
        f"Temperature is {hw.get('temperature', 'unknown')} degrees. "
        f"Movement intensity is {hw.get('movement_intensity', 'unknown')}. "
        f"{pulse_line} "
        f"{ml.get('summary_long', 'No additional ML summary available.')}"
    )

    return summary

summary_text = build_summary()
speak(summary_text)

