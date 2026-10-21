
import cv2
import math
from pathlib import Path

VIDEO_PATH = "data/input/sample.mp4"
OUTPUT_PATH = Path("data/output/camera_inspection/motion_bursts.jpg")

frames_to_inspect = [
    350, 360, 370, 380, 390, 400, 410, 420, 430,
    800, 815, 825, 835, 845, 855, 865, 875, 890
]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

thumb_w = 320
thumb_h = 180
label_h = 28
columns = 3
rows = math.ceil(len(frames_to_inspect) / columns)

sheet = cv2.cvtColor(
    cv2.UMat(1, 1, cv2.CV_8UC3).get(),
    cv2.COLOR_BGR2RGB
)

# Build a blank dark contact sheet.
sheet = (
    __import__("numpy").zeros(
        (rows * (thumb_h + label_h), columns * thumb_w, 3),
        dtype="uint8"
    )
)

for i, frame_number in enumerate(frames_to_inspect):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()

    if not ok:
        print(f"Could not read frame {frame_number}")
        continue

    thumb = cv2.resize(frame, (thumb_w, thumb_h))
    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    sheet[y:y + thumb_h, x:x + thumb_w] = thumb

    cv2.putText(
        sheet,
        f"Frame {frame_number} ({frame_number / 25:.1f}s)",
        (x + 8, y + thumb_h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

cap.release()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), sheet)

print(f"Saved: {OUTPUT_PATH}")
print(f"Frames inspected: {len(frames_to_inspect)}")
