
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
        np.array([90, 255, 255], np.uint8),
    )
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def estimate_flow(a, b):
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
        ransacReprojThreshold=2.0,
    )
    return M


def estimate_orb(a, b):
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
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return M


def alignment_error(a, b, M):
    if M is None:
        return float("nan"), 0

    h, w = a.shape[:2]

    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    aligned = cv2.warpAffine(gb, M, (w, h))

    valid = cv2.warpAffine(
        np.full((h, w), 255, np.uint8),
        M, (w, h),
        flags=cv2.INTER_NEAREST
    ) > 0

    # Evaluate only grass regions visible in both frames.
    mask_a = grass_mask(a) > 0
    mask_b = cv2.warpAffine(
        grass_mask(b), M, (w, h),
        flags=cv2.INTER_NEAREST
    ) > 0

    # Exclude flat areas with little image texture.
    gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1)
    texture = cv2.magnitude(gx, gy) > 15

    mask = valid & mask_a & mask_b & texture

    count = int(mask.sum())
    if count < 500:
        return float("nan"), count

    difference = cv2.absdiff(ga, aligned)
    return float(np.median(difference[mask])), count


print(
    "pair | flow median error | ORB median error | "
    "flow pixels | ORB pixels"
)

for a_idx, b_idx in PAIRS:
    a = read_frame(a_idx)
    b = read_frame(b_idx)

    mf = estimate_flow(a, b)
    mo = estimate_orb(a, b)

    ef, nf = alignment_error(a, b, mf)
    eo, no = alignment_error(a, b, mo)

    print(
        f"{a_idx:4d}-{b_idx:<4d} | "
        f"{ef:17.2f} | "
        f"{eo:16.2f} | "
        f"{nf:11d} | "
        f"{no:10d}"
    )

cap.release()

print("\nAlignment error test complete.")
print("Lower median error generally indicates better image agreement.")
print("These are grayscale intensity errors, not geometric pixel errors.")
