import json
import os
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# -----------------------------
# ARDUINO SUPPORT
# -----------------------------
try:
    from arduino import init_arduino, send_to_arduino
except Exception:  # pragma: no cover - hardware dependency guard
    def init_arduino():
        return None

    def send_to_arduino(message):
        return None

# -----------------------------
# EXISTING BACKEND CODE
# -----------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PULSE_SCRIPT = BACKEND_DIR / "PulseONLINE.py"
ML_SCRIPT = BACKEND_DIR / "CareSenseML.py"
PULSE_OUTPUT = BACKEND_DIR / "pulse.json"
ML_OUTPUT = BACKEND_DIR / "ml_output.json"
PID_FILES = {
    "pulse": BACKEND_DIR / "pulse_camera.pid",
    "ml": BACKEND_DIR / "ml_camera.pid",
}
SCRIPT_MAP = {
    "pulse": PULSE_SCRIPT,
    "ml": ML_SCRIPT,
}
PROCESSES = {
    "pulse": None,
    "ml": None,
}


def load_json_file(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def write_pid(name, pid):
    PID_FILES[name].write_text(str(pid), encoding="utf-8")


def read_pid(name):
    path = PID_FILES[name]
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def remove_pid(name):
    path = PID_FILES[name]
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def is_pid_active(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_process(name):
    proc = PROCESSES.get(name)
    if proc and proc.poll() is None:
        return proc
    PROCESSES[name] = None
    return None


def process_running(name):
    proc = get_process(name)
    if proc:
        return True
    pid = read_pid(name)
    return is_pid_active(pid)


def start_module(name):
    try:
        if process_running(name):
            return {"success": True, "running": True, "status": "online", "message": "Already running."}

        script_path = SCRIPT_MAP[name]
        if not script_path.exists():
            return {"success": False, "running": False, "status": "offline", "message": f"Script not found: {script_path.name}"}

        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        PROCESSES[name] = proc
        write_pid(name, proc.pid)
        return {"success": True, "running": True, "status": "online", "message": "Module started.", "pid": proc.pid}
    except Exception as exc:
        return {"success": False, "running": False, "status": "offline", "message": f"Failed to start module: {exc}"}


def stop_module(name):
    try:
        proc = get_process(name)
        stopped = False
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
            PROCESSES[name] = None
            stopped = True

        pid = read_pid(name)
        if pid and is_pid_active(pid):
            try:
                if os.name == "nt":
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
                stopped = True
            except Exception:
                pass

        remove_pid(name)
        if stopped:
            return {"success": True, "running": False, "status": "offline", "message": "Module stopped."}
        return {"success": True, "running": False, "status": "offline", "message": "Module was not running."}
    except Exception as exc:
        return {"success": False, "running": False, "status": "offline", "message": f"Failed to stop module: {exc}"}


def _is_serious_condition(data):
    if not isinstance(data, dict):
        return False

    candidate_payloads = []
    if isinstance(data.get("prediction"), dict):
        candidate_payloads.append(data["prediction"])
    if isinstance(data.get("result"), dict):
        candidate_payloads.append(data["result"])
    candidate_payloads.append(data)

    for payload in candidate_payloads:
        for key in ("prediction", "event_type", "event", "status"):
            value = payload.get(key)
            if isinstance(value, str) and value.lower() in {"serious", "danger", "critical", "emergency", "abnormal", "abnormal_movement", "stroke_like_asymmetry"}:
                return True

        for key in ("risk_score", "confidence"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                if value > 1.0:
                    return value >= 50.0
                if value > 0.8:
                    return True

    return False


def _update_arduino_alert(data):
    try:
        if _is_serious_condition(data):
            send_to_arduino("ALERT")
        else:
            send_to_arduino("CLEAR")
    except Exception:
        pass


def get_module_status(name):
    try:
        output_file = PULSE_OUTPUT if name == "pulse" else ML_OUTPUT
        result = load_json_file(output_file)
        result["running"] = process_running(name)
        result["last_checked"] = __import__("datetime").datetime.now().isoformat()
        result["status"] = "online" if result["running"] else "offline"
        _update_arduino_alert(result)
        return result
    except Exception:
        return {"status": "offline", "running": False, "last_checked": __import__("datetime").datetime.now().isoformat()}


class ApiHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, body, status=200):
        self._set_headers(status)
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/pulse":
            self._send_json(get_module_status("pulse"))
            return
        if path == "/api/ml":
            self._send_json(get_module_status("ml"))
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # -----------------------------
        # ARDUINO ENDPOINT
        # -----------------------------
        if path == "/api/arduino/send":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body) if body else {}
            except Exception:
                data = {}

            try:
                msg = data.get("message")
                if msg:
                    send_to_arduino(msg)
                    self._send_json({"success": True, "sent": msg})
                else:
                    self._send_json({"error": "No message provided"}, status=400)
            except Exception:
                self._send_json({"error": "Failed to send Arduino command"}, status=500)
            return

        # Existing endpoints
        if path == "/api/pulse/start":
            self._send_json(start_module("pulse"))
            return
        if path == "/api/pulse/stop":
            self._send_json(stop_module("pulse"))
            return
        if path == "/api/ml/start":
            self._send_json(start_module("ml"))
            return
        if path == "/api/ml/stop":
            self._send_json(stop_module("ml"))
            return

        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format, *args):
        return


def run_server(host="127.0.0.1", port=8000):
    try:
        init_arduino()
    except Exception:
        pass

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ApiHandler)
    print(f"Backend API running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
