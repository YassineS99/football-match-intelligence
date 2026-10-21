
import cv2
import numpy as np

VIDEO_PATH = "data/input/sample.mp4"

PAIRS = [
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
        raise RuntimeError(f"Cannot read frame {index}")
    return frame


def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([30, 45, 35], np.uint8),
        np.array([90, 255, 255], np.uint8),
    )
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def optical_flow_transform(a, b):
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    pts = cv2.goodFeaturesToTrack(
        gray_a, 600, 0.005, 8,
        mask=grass_mask(a)
    )

    if pts is None or len(pts) < 10:
        return None, 0

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, pts, None
    )

    if nxt is None or status is None:
        return None, 0

    good = status.ravel() == 1

    if good.sum() < 10:
        return None, 0

    M, inlier_mask = cv2.estimateAffinePartial2D(
        nxt[good],
        pts[good],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )

    count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return M, count


def orb_transform(a, b):
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    kp_a, des_a = orb.detectAndCompute(
        gray_a, grass_mask(a)
    )
    kp_b, des_b = orb.detectAndCompute(
        gray_b, grass_mask(b)
    )

    if des_a is None or des_b is None:
        return None, 0

    pairs = matcher.knnMatch(des_b, des_a, k=2)

    good = [
        m for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 8:
        return None, 0

    src = np.float32([
        kp_b[m.queryIdx].pt for m in good
    ])

    dst = np.float32([
        kp_a[m.trainIdx].pt for m in good
    ])

    M, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99,
    )

    count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return M, count


def transform_points(M, points):
    return cv2.transform(
        points.reshape(1, -1, 2), M
    ).reshape(-1, 2)


print(
    "pair | flow inliers | ORB inliers | "
    "mean disagreement px | max disagreement px"
)

for a_idx, b_idx in PAIRS:
    a = get_frame(a_idx)
    b = get_frame(b_idx)

    M_flow, n_flow = optical_flow_transform(a, b)
    M_orb, n_orb = orb_transform(a, b)

    if M_flow is None or M_orb is None:
        print(
            f"{a_idx}-{b_idx} | {n_flow} | {n_orb} | "
            "cannot compare"
        )
        continue

    h, w = a.shape[:2]

    points = np.float32([
        [w * 0.25, h * 0.25],
        [w * 0.75, h * 0.25],
        [w * 0.50, h * 0.50],
        [w * 0.25, h * 0.75],
        [w * 0.75, h * 0.75],
    ])

    p_flow = transform_points(M_flow, points)
    p_orb = transform_points(M_orb, points)

    errors = np.linalg.norm(
        p_flow - p_orb, axis=1
    )

    print(
        f"{a_idx:4d}-{b_idx:<4d} | "
        f"{n_flow:12d} | "
        f"{n_orb:11d} | "
        f"{errors.mean():20.2f} | "
        f"{errors.max():19.2f}"
    )

cap.release()

print("\nComparison complete.")
print("No existing files were modified.")
