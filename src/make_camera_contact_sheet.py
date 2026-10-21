
import cv2
import math
from pathlib import Path

INPUT_DIR = Path("data/output/camera_inspection")
OUTPUT_PATH = INPUT_DIR / "camera_contact_sheet.jpg"

images = sorted(INPUT_DIR.glob("frame_*.jpg"))

if not images:
    raise RuntimeError(f"No frame images found in {INPUT_DIR}")

thumb_width = 320
thumb_height = 180
label_height = 30
columns = 4
rows = math.ceil(len(images) / columns)

sheet = cv2.imread(str(images[0]))
if sheet is None:
    raise RuntimeError("Could not read the first image.")

sheet = cv2.resize(sheet, (thumb_width, thumb_height))
canvas = cv2.imread(str(images[0]))
canvas = cv2.resize(canvas, (thumb_width, thumb_height))
canvas = cv2.copyMakeBorder(
    canvas, 0, label_height, 0, 0,
    cv2.BORDER_CONSTANT, value=(30, 30, 30)
)

canvas_height = rows * (thumb_height + label_height)
canvas_width = columns * thumb_width

contact_sheet = cv2.copyMakeBorder(
    canvas, 0, canvas_height - canvas.shape[0],
    0, canvas_width - canvas.shape[1],
    cv2.BORDER_CONSTANT, value=(30, 30, 30)
)

for i, path in enumerate(images):
    image = cv2.imread(str(path))
    if image is None:
        continue

    image = cv2.resize(image, (thumb_width, thumb_height))
    x = (i % columns) * thumb_width
    y = (i // columns) * (thumb_height + label_height)

    contact_sheet[y:y + thumb_height, x:x + thumb_width] = image

    cv2.putText(
        contact_sheet,
        path.stem.replace("frame_", "Frame "),
        (x + 8, y + thumb_height + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

cv2.imwrite(str(OUTPUT_PATH), contact_sheet)
print(f"Created contact sheet: {OUTPUT_PATH}")
print(f"Frames included: {len(images)}")
