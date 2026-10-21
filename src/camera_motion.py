import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_roles.csv"
OUT_PATH = "data/output/tracks_ref.csv"
CHECK_PATH = "data/output/stitch_check.jpg"
STITCH_FRAMES = [0, 300, 600, 900]

df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}


def background_mask(shape, frame_idx):
    mask = np.full(shape, 255, np.uint8)
    if frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1 = max(int(r.x1) - 10, 0), max(int(r.y1) - 10, 0)
            x2, y2 = int(r.x2) + 10, int(r.y2) + 10
            mask[y1:y2, x1:x2] = 0
    return mask


cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Hs = [np.eye(3)]  # Hs[i] maps frame i -> frame 0
kept = {0: frame} if 0 in STITCH_FRAMES else {}
fallbacks = 0
idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H = None
    pts = cv2.goodFeaturesToTrack(prev_gray, 800, 0.01, 10,
                                  mask=background_mask(prev_gray.shape, idx - 1))
    if pts is not None and len(pts) >= 10:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
        good = st.ravel() == 1
        if good.sum() >= 10:
            H, _ = cv2.findHomography(nxt[good], pts[good], cv2.RANSAC, 3.0)  # cur -> prev
    if H is None:
        H = np.eye(3)
        fallbacks += 1
    new = Hs[-1] @ H
    Hs.append(new / new[2, 2])
    if idx in STITCH_FRAMES:
        kept[idx] = frame
    prev_gray = gray
cap.release()

# Player foot positions in the shared (frame 0) view
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = df["y2"]
ref_x = np.zeros(len(df))
ref_y = np.zeros(len(df))
for f, g in df.groupby("frame"):
    pts = g[["cx", "cy"]].to_numpy(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, Hs[f]).reshape(-1, 2)
    ref_x[df.index.get_indexer(g.index)] = out[:, 0]
    ref_y[df.index.get_indexer(g.index)] = out[:, 1]
df["ref_x"], df["ref_y"] = ref_x, ref_y
df.to_csv(OUT_PATH, index=False)

# Stitch check: warp a few frames into one canvas
h, w = next(iter(kept.values())).shape[:2]
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
all_c = np.vstack([cv2.perspectiveTransform(corners, Hs[f]).reshape(-1, 2) for f in kept])
minx, miny = all_c.min(axis=0)
maxx, maxy = all_c.max(axis=0)
s = min(1.0, 1800 / max(maxx - minx, maxy - miny))
T = np.array([[s, 0, -s * minx], [0, s, -s * miny], [0, 0, 1]])
cw, ch = int((maxx - minx) * s), int((maxy - miny) * s)
acc = np.zeros((ch, cw, 3), np.float32)
cnt = np.zeros((ch, cw), np.float32)
for f, img in kept.items():
    M = T @ Hs[f]
    acc += cv2.warpPerspective(img, M, (cw, ch)).astype(np.float32)
    cnt += cv2.warpPerspective(np.ones((h, w), np.float32), M, (cw, ch))
canvas = (acc / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
cv2.imwrite(CHECK_PATH, canvas)

print("Frames processed:", idx + 1)
print("Frames where motion estimate failed (fallback):", fallbacks)
print("Stitched canvas size:", cw, "x", ch)
print("Saved:", OUT_PATH)
print("Saved:", CHECK_PATH)