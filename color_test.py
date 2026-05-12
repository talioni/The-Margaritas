import cv2
import numpy as np

# Editable HSV ranges for this prototype. Tune these values for your camera/light.
COLOR_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((36, 60, 40), (86, 255, 255))],
    "blue": [((90, 80, 40), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "brown": [((10, 80, 20), (20, 255, 200))],
    "white": [((0, 0, 200), (180, 40, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
}


def detect_dominant_color(hsv_roi):
    pixel_counts = {}

    for color_name, hsv_ranges in COLOR_RANGES.items():
        total_pixels_for_color = 0
        for lower, upper in hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower_np, upper_np)
            total_pixels_for_color += cv2.countNonZero(mask)
        pixel_counts[color_name] = total_pixels_for_color

    dominant_color, dominant_count = max(pixel_counts.items(), key=lambda item: item[1])
    roi_pixel_count = hsv_roi.shape[0] * hsv_roi.shape[1]

    if dominant_count < roi_pixel_count * 0.05:
        return "unknown"

    return dominant_color


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default webcam (index 0).")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from webcam.")
                break

            frame_height, frame_width = frame.shape[:2]
            roi_size = min(frame_width, frame_height) // 3
            x1 = (frame_width - roi_size) // 2
            y1 = (frame_height - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            detected_color = detect_dominant_color(hsv_roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Detected: {detected_color.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Later this ROI step can feed texture/material features or an ML model.
            cv2.imshow("Color Prototype (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
