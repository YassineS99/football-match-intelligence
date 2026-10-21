
import cv2
import numpy as np

VIDEO_PATH = "data/input/sample.mp4"

TEST_PAIRS = [
    (100, 101),
    (250, 251),
    (375, 376),
    (380, 381),
    (390, 391),
    (400, 401),
    (500, 501),
    (835, 836),
    (840, 841),
    (850, 851),
    (875, 876),
]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def get_frame(index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {index}")
    return frame

def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([30, 45, 35], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8)
    )

    return cv2.erode(mask, np.ones((5, 5), np.uint8))

print(
    "pair | keypoints A | keypoints B | "
    "good matches | inliers | ratio | translation | scale"
)

for a, b in TEST_PAIRS:
    frame_a = get_frame(a)
    frame_b = get_frame(b)

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    kp_a, des_a = orb.detectAndCompute(
        gray_a, grass_mask(frame_a)
    )
    kp_b, des_b = orb.detectAndCompute(
        gray_b, grass_mask(frame_b)
    )

    if des_a is None or des_b is None:
        print(f"{a}-{b}: insufficient descriptors")
        continue

    pairs = matcher.knnMatch(des_b, des_a, k=2)

    good = [
        m for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 8:
        print(
            f"{a}-{b}: only {len(good)} good matches"
        )
        continue

    src = np.float32([
        kp_b[m.queryIdx].pt for m in good
    ])

    dst = np.float32([
        kp_a[m.trainIdx].pt for m in good
    ])

    M, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99
    )

    if M is None or mask is None:
        print(f"{a}-{b}: estimation failed")
        continue

    inliers = int(mask.sum())
    ratio = inliers / len(good)

    translation = float(np.linalg.norm(M[:, 2]))
    scale = float(np.hypot(M[0, 0], M[1, 0]))

    print(
        f"{a:4d}-{b:<4d} | "
        f"{len(kp_a):11d} | "
        f"{len(kp_b):11d} | "
        f"{len(good):12d} | "
        f"{inliers:7d} | "
        f"{ratio:5.2f} | "
        f"{translation:11.2f} | "
        f"{scale:5.3f}"
    )

cap.release()

print("\nORB diagnostic completed.")
print("Existing project outputs were not modified.")
