
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

VIDEO = "data/input/sample.mp4"
OUTPUT = Path("data/output")
OUTPUT.mkdir(parents=True, exist_ok=True)

orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([30, 45, 35], np.uint8),
        np.array([90, 255, 255], np.uint8)
    )
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def estimate_flow(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    pts = cv2.goodFeaturesToTrack(
        ga, 600, 0.005, 8, mask=grass_mask(a)
    )

    if pts is None or len(pts) < 10:
        return None, 0, 0.0

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        ga, gb, pts, None
    )

    if nxt is None or status is None:
        return None, 0, 0.0

    good = status.ravel() == 1

    if good.sum() < 10:
        return None, 0, 0.0

    M, mask = cv2.estimateAffinePartial2D(
        nxt[good], pts[good],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0
    )

    if M is None or mask is None:
        return None, 0, 0.0

    n = int(mask.sum())
    return M, n, n / int(good.sum())


def estimate_orb(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    ka, da = orb.detectAndCompute(ga, grass_mask(a))
    kb, db = orb.detectAndCompute(gb, grass_mask(b))

    if da is None or db is None:
        return None, 0, 0.0

    pairs = matcher.knnMatch(db, da, k=2)

    good = [
        m for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 8:
        return None, 0, 0.0

    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])

    M, mask = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000
    )

    if M is None or mask is None:
        return None, 0, 0.0

    n = int(mask.sum())
    return M, n, n / len(good)


def plausible(M):
    if M is None or not np.isfinite(M).all():
        return False

    scale = np.hypot(M[0, 0], M[1, 0])
    rotation = np.degrees(
        np.arctan2(M[1, 0], M[0, 0])
    )

    return 0.94 <= scale <= 1.06 and abs(rotation) <= 5


def acceptable(M, n, ratio):
    return (
        plausible(M)
        and n >= 40
        and ratio >= 0.50
    )


cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

ok, previous = cap.read()

if not ok:
    raise RuntimeError("Cannot read first frame")

transforms = [np.eye(3, dtype=np.float64)]
records = [{
    "frame": 0,
    "method": "reference",
    "inliers": 0,
    "ratio": 1.0,
    "valid": True,
    "scale": 1.0,
    "translation": 0.0
}]

frame_idx = 0

while True:
    ok, current = cap.read()

    if not ok:
        break

    frame_idx += 1

    mf, nf, rf = estimate_flow(previous, current)

    method = "flow"
    M, n, ratio = mf, nf, rf

    # ORB is evaluated only when flow is unreliable.
    if not acceptable(M, n, ratio):
        mo, no, ro = estimate_orb(previous, current)

        if acceptable(mo, no, ro):
            M, n, ratio = mo, no, ro
            method = "orb"
        else:
            M = None
            method = "failed"

    valid = M is not None

    if valid:
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M

        scale = float(np.hypot(M[0, 0], M[1, 0]))
        translation = float(np.linalg.norm(M[:, 2]))
    else:
        H = np.full((3, 3), np.nan)
        scale = float("nan")
        translation = float("nan")

    transforms.append(H)

    records.append({
        "frame": frame_idx,
        "method": method,
        "inliers": n,
        "ratio": ratio,
        "valid": valid,
        "scale": scale,
        "translation": translation
    })

    if frame_idx % 100 == 0:
        print(f"Processed frame {frame_idx}")

    previous = current

cap.release()

np.save(
    OUTPUT / "camera_transforms_hybrid.npy",
    np.array(transforms)
)

df = pd.DataFrame(records)
df.to_csv(
    OUTPUT / "camera_quality_hybrid.csv",
    index=False
)

print("\nHybrid camera estimation complete.")
print("\nMethod counts:")
print(df["method"].value_counts())

print("\nValid frame percentage:")
print(f"{df['valid'].mean() * 100:.1f}%")

print("\nSaved:")
print("data/output/camera_transforms_hybrid.npy")
print("data/output/camera_quality_hybrid.csv")
