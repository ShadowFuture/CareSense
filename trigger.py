try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None


def send_trigger_request(url="http://localhost:5000/ml/analyze/webcam"):
    if requests is None:
        print("requests package is not installed; skipping trigger request")
        return None

    try:
        response = requests.post(url)
        print("Server response:")
        print(response.text)
        return response
    except Exception as e:
        print("Error:", e)
        return None


if __name__ == "__main__":
    send_trigger_request()
