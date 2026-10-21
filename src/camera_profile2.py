import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_roles.csv"
HS_PATH = "data/output/homographies_v2.npy"

df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}


def pitch_mask(frame, frame_idx):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)).astype(np.uint8) * 255
    grass = cv2.erode(grass, np.ones((9, 9), np.uint8))
    if frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1 = max(int(r.x1) - 10, 0), max(int(r.y1) - 10, 0)
            grass[y1:int(r.y2) + 10, x1:int(r.x2) + 10] = 0
    return grass


cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ok, frame = cap.read()
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_mask = pitch_mask(frame, 0)
Hs = [np.eye(3)]
inliers = [0]
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    M, n_in = None, 0
    pts = cv2.goodFeaturesToTrack(prev, 600, 0.005, 8, mask=prev_mask)
    if pts is not None and len(pts) >= 10:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = st.ravel() == 1
        if good.sum() >= 10:
            # similarity transform current -> previous (scale, rotation, shift)
            M, inl = cv2.estimateAffinePartial2D(nxt[good], pts[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0)
            n_in = int(inl.sum()) if inl is not None else 0
    H = np.eye(3)
    if M is not None:
        H[:2, :] = M
    Hs.append(Hs[-1] @ H)
    inliers.append(n_in)
    prev, prev_mask = gray, pitch_mask(frame, idx)
cap.release()
np.save(HS_PATH, np.array(Hs))

c = np.array([[w / 2, h / 2], [w / 2 + 100, h / 2]], np.float32).reshape(-1, 1, 2)
print("frame | zoom vs frame 0 | center in ref view (x, y) | min inliers (last 50)")
for f in range(0, len(Hs), 50):
    p = cv2.perspectiveTransform(c, Hs[f]).reshape(-1, 2)
    zoom = np.linalg.norm(p[1] - p[0]) / 100
    lo = min(inliers[max(f - 49, 1):f + 1]) if f > 0 else 0
    print(f"{f:5d} | {zoom:6.2f}          | ({p[0][0]:7.0f}, {p[0][1]:7.0f})        | {lo}")
print("\nSaved:", HS_PATH)