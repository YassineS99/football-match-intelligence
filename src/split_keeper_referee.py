import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_teams.csv"
MONTAGE_PATH = "data/output/keeper_vs_referee.jpg"
SAMPLE_EVERY = 5
MIN_SAMPLES = 3
TOP_N = 20


def region(frame, x1, y1, x2, y2, top, bottom):
    h, w = y2 - y1, x2 - x1
    ry1, ry2 = int(y1 + top * h), int(y1 + bottom * h)
    rx1, rx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    return frame[max(ry1, 0):ry2, max(rx1, 0):rx2]


def grass_frac(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)).mean())


def dark_frac(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float((hsv[..., 2] < 70).mean())


df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}

samples, crops = [], {}
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % SAMPLE_EVERY == 0 and frame_idx in by_frame:
        for r in by_frame[frame_idx].itertuples():
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            torso = region(frame, x1, y1, x2, y2, 0.2, 0.55)
            shorts = region(frame, x1, y1, x2, y2, 0.58, 0.78)
            if torso.size == 0 or shorts.size == 0:
                continue
            samples.append((r.track_id, grass_frac(torso), dark_frac(shorts)))
            if r.track_id not in crops:
                body = frame[max(y1, 0):y2, max(x1, 0):x2]
                if body.size > 0:
                    crops[r.track_id] = cv2.resize(body, (60, 100))
    frame_idx += 1
cap.release()

feats = pd.DataFrame(samples, columns=["track_id", "grass", "dark_shorts"])
stats = feats.groupby("track_id").agg(
    grass=("grass", "median"), dark_shorts=("dark_shorts", "median"), n=("grass", "size")
)
stats = stats[stats["n"] >= MIN_SAMPLES].sort_values("grass", ascending=False).head(TOP_N)

print("Top tracks by torso grass fraction (dark_shorts high = likely referee):")
for tid, row in stats.iterrows():
    print(f"  track {tid}: grass={row['grass']:.2f}, dark_shorts={row['dark_shorts']:.2f}, samples={int(row['n'])}")

per_row = 10
canvas = np.zeros((2 * 135, per_row * 60, 3), dtype=np.uint8)
for k, tid in enumerate(stats.index):
    r, c = divmod(k, per_row)
    y0, x0 = r * 135, c * 60
    if tid in crops:
        canvas[y0 + 35:y0 + 135, x0:x0 + 60] = crops[tid]
    cv2.putText(canvas, f"#{tid}", (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f"d={stats.loc[tid, 'dark_shorts']:.2f}", (x0 + 2, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(MONTAGE_PATH, canvas)
print("\nSaved:", MONTAGE_PATH)