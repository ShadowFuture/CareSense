import cv2
import numpy as np

def calculate_image_confidence(frame):
    """
    Computes a dynamic confidence multiplier (0.0 to 1.0) based on frame quality.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Measure Sharpness via Laplacian Variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 150.0)  # Normalize to max 1.0

    # 2. Measure Lighting Quality
    mean_brightness = np.mean(gray)
    if mean_brightness < 40 or mean_brightness > 220:
        lighting_score = 0.5  # Too dark or overexposed
    else:
        lighting_score = 1.0  # Good lighting conditions

    # Combine metrics into a single confidence factor
    image_quality_confidence = float(blur_score * lighting_score)
    return image_quality_confidence


def run_ml_inference(accel_impact, post_impact_movement, facial_asymmetry, heart_rate, frame=None):
    # Base model calculation
    if facial_asymmetry > 0.4:
        event_type = "stroke_like_asymmetry"
        risk_score = 78
        base_confidence = 0.90
    else:
        event_type = "normal"
        risk_score = 10
        base_confidence = 0.95

    # Adjust confidence dynamically if an image/frame was provided
    if frame is not None:
        quality_factor = calculate_image_confidence(frame)
        final_confidence = round(base_confidence * quality_factor, 2)
    else:
        final_confidence = base_confidence

    return {
        "event_type": event_type,
        "risk_score": risk_score,
        "confidence": final_confidence,
    }


# ==========================================
# CALLING THE FUNCTIONS TO RUN THE CODE
# ==========================================
if __name__ == "__main__":
    # Create a dummy image frame (480x640 random pixels representing a camera frame)
    sample_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Alternatively, load a real image using OpenCV:
    # sample_frame = cv2.imread("your_image.jpg")

    # Call the ML inference function with sample test data
    results = run_ml_inference(
        accel_impact=2.5,
        post_impact_movement=0.1,
        facial_asymmetry=0.5, # > 0.4 will trigger stroke_like_asymmetry
        heart_rate=85,
        frame=sample_frame
    )

    # Print output to terminal
    print("Inference Output:")
    print(results)
