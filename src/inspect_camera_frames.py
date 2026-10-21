
import cv2
from pathlib import Path

VIDEO_PATH = "data/input/sample.mp4"
OUTPUT_DIR = Path("data/output/camera_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames: {total_frames}")
print(f"FPS: {fps:.2f}")
print(f"Duration: {total_frames / fps:.2f} seconds" if fps > 0 else "FPS unavailable")

# Inspect the transition where the camera estimates became unstable.
frames_to_inspect = [0, 50, 100, 150, 200, 250, 275, 300, 325, 350, 375, 400, 450, 500, 700, 1000]

for frame_number in frames_to_inspect:
    if frame_number >= total_frames:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()

    if not ok:
        print(f"Could not read frame {frame_number}")
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

    cv2.imwrite(str(output_path), frame)
    print(f"Saved: {output_path}")

cap.release()

print(f"\nInspection images saved in: {OUTPUT_DIR}")
