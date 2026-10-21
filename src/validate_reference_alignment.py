
import cv2
import numpy as np
from pathlib import Path

VIDEO = "data/input/sample.mp4"
TRANSFORMS = "data/output/camera_transforms_hybrid.npy"
OUTPUT = Path("data/output/reference_alignment")
OUTPUT.mkdir(parents=True, exist_ok=True)

TEST_FRAMES = [25, 50, 100, 150, 200]

Hs = np.load(TRANSFORMS)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

ok, reference = cap.read()

if not ok:
    raise RuntimeError("Cannot read reference frame")

height, width = reference.shape[:2]


def read_frame(index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")

    return frame


def cumulative_transform(frame_number):
    H = np.eye(3, dtype=np.float64)

    for f in range(1, frame_number + 1):
        step = Hs[f]

        if not np.isfinite(step).all():
            raise RuntimeError(
                f"Invalid transform at frame {f}"
            )

        H = H @ step

    return H


for frame_number in TEST_FRAMES:
    frame = read_frame(frame_number)

    H = cumulative_transform(frame_number)

    aligned = cv2.warpPerspective(
        frame,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR
    )

    valid = cv2.warpPerspective(
        np.full((height, width), 255, np.uint8),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST
    ) > 0

    # Red channel: reference frame.
    # Green channel: transformed current frame.
    # Correctly aligned stationary structures
    # should appear approximately yellow.
    ref_gray = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )

    aligned_gray = cv2.cvtColor(
        aligned, cv2.COLOR_BGR2GRAY
    )

    overlay = np.zeros_like(reference)

    overlay[:, :, 2] = ref_gray
    overlay[:, :, 1] = aligned_gray

    overlay[~valid] = 0

    cv2.putText(
        overlay,
        f"Frame 0 vs {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    output_path = OUTPUT / f"alignment_{frame_number:04d}.jpg"

    cv2.imwrite(str(output_path), overlay)

    print(f"Saved: {output_path}")

cap.release()

print("\nReference alignment validation complete.")
