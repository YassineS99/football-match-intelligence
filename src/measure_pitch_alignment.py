
import cv2
import numpy as np
from pathlib import Path

VIDEO = "data/input/sample.mp4"
TRANSFORMS = "data/output/camera_transforms_hybrid.npy"

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


def pitch_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Approximate green pitch mask.
    grass = cv2.inRange(
        hsv,
        np.array([35, 40, 30]),
        np.array([90, 255, 255])
    )

    # Bright, low-saturation white markings.
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 125]),
        np.array([179, 100, 255])
    )

    # Restrict white pixels to areas near grass.
    nearby_grass = cv2.dilate(
        grass,
        np.ones((11, 11), np.uint8)
    )

    mask = cv2.bitwise_and(white, nearby_grass)

    return mask


def cumulative_transform(frame_number):
    H = np.eye(3, dtype=np.float64)

    for f in range(1, frame_number + 1):
        H = H @ Hs[f]

    return H


reference_lines = pitch_lines(reference)

distance_map = cv2.distanceTransform(
    255 - reference_lines,
    cv2.DIST_L2,
    3
)

print("frame | matched pixels | median distance | p90 distance")
print("-" * 65)

for f in TEST_FRAMES:
    frame = read_frame(f)

    H = cumulative_transform(f)

    lines = pitch_lines(frame)

    warped = cv2.warpPerspective(
        lines,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

    valid = cv2.warpPerspective(
        np.full((height, width), 255, np.uint8),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST
    ) > 0

    selected = (warped > 0) & valid

    distances = distance_map[selected]

    if len(distances) == 0:
        print(f"{f:5d} | no usable pixels")
        continue

    median = np.median(distances)
    p90 = np.percentile(distances, 90)

    print(
        f"{f:5d} | "
        f"{len(distances):14d} | "
        f"{median:15.2f} | "
        f"{p90:12.2f}"
    )

cap.release()

print("\nPitch alignment diagnostic complete.")
print("Note: White-player pixels and other objects can contaminate results.")
