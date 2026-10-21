import cv2

VIDEO_PATH = "data/input/sample.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps else 0

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {frame_count}")
print(f"Duration: {duration:.1f} seconds")

cap.release()