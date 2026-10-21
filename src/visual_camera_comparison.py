
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/input/sample.mp4"
OUTPUT_DIR = Path("data/output/camera_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [(375, 376), (380, 381), (840, 841), (850, 851)]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def read_frame(index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")
    return frame


def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([30, 45, 35], np.uint8),
        np.array([90, 255, 255], np.uint8)
    )
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def flow_transform(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    pts = cv2.goodFeaturesToTrack(
        ga, 600, 0.005, 8, mask=grass_mask(a)
    )
    if pts is None or len(pts) < 10:
        return None

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        ga, gb, pts, None
    )
    if nxt is None or status is None:
        return None

    valid = status.ravel() == 1
    if valid.sum() < 10:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        nxt[valid], pts[valid],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0
    )
    return M


def orb_transform(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    ka, da = orb.detectAndCompute(ga, grass_mask(a))
    kb, db = orb.detectAndCompute(gb, grass_mask(b))

    if da is None or db is None:
        return None

    matches = matcher.knnMatch(db, da, k=2)
    good = [
        m for pair in matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 8:
        return None

    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])

    M, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    return M


def make_overlay(reference, moving, M, label):
    h, w = reference.shape[:2]

    if M is None:
        result = np.zeros_like(reference)
        cv2.putText(
            result, "TRANSFORM FAILED", (40, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2
        )
    else:
        aligned = cv2.warpAffine(
            moving, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        valid = cv2.warpAffine(
            np.full((h, w), 255, np.uint8),
            M, (w, h),
            flags=cv2.INTER_NEAREST
        )

        result = cv2.addWeighted(
            reference, 0.5, aligned, 0.5, 0
        )
        result[valid == 0] = reference[valid == 0]

    cv2.putText(
        result, label, (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 255, 255), 2
    )
    return result


for a_idx, b_idx in PAIRS:
    a = read_frame(a_idx)
    b = read_frame(b_idx)

    mf = flow_transform(a, b)
    mo = orb_transform(a, b)

    left = make_overlay(a, b, mf, "Optical Flow")
    right = make_overlay(a, b, mo, "ORB")

    comparison = np.hstack([left, right])

    path = OUTPUT_DIR / f"compare_{a_idx}_{b_idx}.jpg"
    cv2.imwrite(str(path), comparison)

    print(f"Saved: {path}")

cap.release()
print("\nVisual comparisons complete.")
