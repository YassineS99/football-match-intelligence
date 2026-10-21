import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_teams.csv"
MONTAGE_PATH = "data/output/grass_fraction_top.jpg"
SAMPLE_EVERY = 5
MIN_SAMPLES = 3
TOP_N = 10


def grass_fraction(frame, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = int(y1 + 0.2 * h), int(y1 + 0.55 * h)
    tx1, tx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    crop = frame[max(ty1, 0):ty2, max(tx1, 0):tx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)
    return float(grass.mean())


df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}
track_cluster = df.groupby("track_id")["cluster"].first()

samples = []
crops = {}

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % SAMPLE_EVERY == 0 and frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            frac = grass_fraction(frame, x1, y1, x2, y2)
            if frac is None:
                continue
            samples.append((r.track_id, frac))
            if r.track_id not in crops:
                body = frame[max(y1, 0):y2, max(x1, 0):x2]
                if body.size > 0:
                    crops[r.track_id] = cv2.resize(body, (60, 100))
    frame_idx += 1
cap.release()

feats = pd.DataFrame(samples, columns=["track_id", "grass"])
stats = feats.groupby("track_id")["grass"].agg(["median", "size"])
stats = stats[stats["size"] >= MIN_SAMPLES].sort_values("median", ascending=False)

print("Median grass fraction across all tracks:", round(stats["median"].median(), 2))
print()
print("Top tracks by grass fraction:")
for tid, row in stats.head(TOP_N).iterrows():
    print(f"  track {tid}: grass={row['median']:.2f}, samples={int(row['size'])}, cluster={track_cluster[tid]}")

canvas = np.zeros((130, TOP_N * 60, 3), dtype=np.uint8)
for k, tid in enumerate(stats.head(TOP_N).index):
    if tid in crops:
        canvas[30:130, k * 60:(k + 1) * 60] = crops[tid]
    cv2.putText(canvas, f"#{tid}", (k * 60 + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(MONTAGE_PATH, canvas)
print("\nSaved:", MONTAGE_PATH)