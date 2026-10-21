import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_roles.csv"
CHECK_FRAMES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
STEP = 25  # direct comparison against frame f - STEP... chained over STEP frames only

df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}


def background_mask(shape, frame_idx):
    mask = np.full(shape, 255, np.uint8)
    if frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1 = max(int(r.x1) - 10, 0), max(int(r.y1) - 10, 0)
            mask[y1:int(r.y2) + 10, x1:int(r.x2) + 10] = 0
    return mask


def pair_homography(a, b, fa):
    """Homography mapping frame b coords -> frame a coords using SIFT matches."""
    sift = cv2.SIFT_create(3000)
    ka, da = sift.detectAndCompute(a, background_mask(a.shape, fa))
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for m, n in (x for x in matches if len(x) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < 15:
        return None, len(good)
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(inl.sum()) if inl is not None else 0


def read_gray(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if ok else None


cap = cv2.VideoCapture(VIDEO_PATH)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
probe = np.float32([[w * 0.25, h * 0.7], [w * 0.5, h * 0.7], [w * 0.75, h * 0.7]]).reshape(-1, 1, 2)

# Chain the pairwise (frame i -> frame i-STEP) homographies from frame 0 up to each check frame,
# using a coarser step than 1 frame so far fewer errors accumulate than in camera_motion.py.
print("Chained-by-25-frames vs chained-by-1-frame positions of 3 probe points (pixel gap):\n")
H_chain = np.eye(3)
prev_idx = 0
prev_gray = read_gray(cap, 0)
results = {}
for target in range(STEP, 1050, STEP):
    g = read_gray(cap, target)
    if g is None:
        break
    H, n = pair_homography(prev_gray, g, prev_idx)
    if H is None:
        print(f"frame {target}: not enough matches, stopping")
        break
    H_chain = H_chain @ H
    H_chain /= H_chain[2, 2]
    results[target] = H_chain.copy()
    prev_gray, prev_idx = g, target
cap.release()

# Recompute the 1-frame chain from camera_motion.py's logic for comparison
cap = cv2.VideoCapture(VIDEO_PATH)
ok, fr = cap.read()
pg = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
Hs = [np.eye(3)]
idx = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    idx += 1
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(pg, 800, 0.01, 10, mask=background_mask(pg.shape, idx - 1))
    H = None
    if pts is not None and len(pts) >= 10:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, gr, pts, None)
        ok_pts = st.ravel() == 1
        if ok_pts.sum() >= 10:
            H, _ = cv2.findHomography(nxt[ok_pts], pts[ok_pts], cv2.RANSAC, 3.0)
    if H is None:
        H = np.eye(3)
    n_ = Hs[-1] @ H
    Hs.append(n_ / n_[2, 2])
    pg = gr
cap.release()

for target in sorted(results):
    if target % 100 != 0 and target not in (25, 50, 75):
        continue
    a = cv2.perspectiveTransform(probe, results[target]).reshape(-1, 2)
    b = cv2.perspectiveTransform(probe, Hs[target]).reshape(-1, 2)
    gap = np.linalg.norm(a - b, axis=1).mean()
    print(f"frame {target:4d}: mean gap = {gap:7.1f} px")