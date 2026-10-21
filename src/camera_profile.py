import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_roles.csv"
HS_PATH = "data/output/homographies.npy"

df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}


def background_mask(shape, frame_idx):
    mask = np.full(shape, 255, np.uint8)
    if frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1 = max(int(r.x1) - 10, 0), max(int(r.y1) - 10, 0)
            mask[y1:int(r.y2) + 10, x1:int(r.x2) + 10] = 0
    return mask


cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ok, frame = cap.read()
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Hs = [np.eye(3)]
inliers = [0]
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, n_in = None, 0
    pts = cv2.goodFeaturesToTrack(prev, 800, 0.01, 10, mask=background_mask(prev.shape, idx - 1))
    if pts is not None and len(pts) >= 10:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = st.ravel() == 1
        if good.sum() >= 10:
            H, inl = cv2.findHomography(nxt[good], pts[good], cv2.RANSAC, 3.0)
            n_in = int(inl.sum()) if inl is not None else 0
    if H is None:
        H = np.eye(3)
    new = Hs[-1] @ H
    Hs.append(new / new[2, 2])
    inliers.append(n_in)
    prev = gray
cap.release()
np.save(HS_PATH, np.array(Hs))

c = np.array([[w / 2, h / 2], [w / 2 + 100, h / 2], [w / 2, h / 2 + 100]], np.float32).reshape(-1, 1, 2)


def local_scale_and_center(H):
    p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    sx = np.linalg.norm(p[1] - p[0]) / 100
    sy = np.linalg.norm(p[2] - p[0]) / 100
    return (sx + sy) / 2, p[0]


print("frame | zoom vs frame 0 | center in ref view (x, y) | pan px/frame (last 50) | min inliers")
prev_center = None
for f in range(0, len(Hs), 50):
    s, ctr = local_scale_and_center(Hs[f])
    pan = ""
    if prev_center is not None:
        pan = f"{np.linalg.norm(ctr - prev_center) / 50:6.1f}"
    lo = min(inliers[max(f - 49, 1):f + 1]) if f > 0 else 0
    print(f"{f:5d} | {s:6.2f}          | ({ctr[0]:7.0f}, {ctr[1]:7.0f})        | {pan:>8}               | {lo}")
    prev_center = ctr
print("\nSaved:", HS_PATH)