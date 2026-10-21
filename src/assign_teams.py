import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_filtered.csv"
OUT_PATH = "data/output/tracks_teams.csv"
MONTAGE_PATH = "data/output/cluster_samples.jpg"
SAMPLE_EVERY = 5
N_CLUSTERS = 4
MIN_SAMPLES = 3  # min color samples a track needs to be clustered


def torso_color(frame, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = int(y1 + 0.2 * h), int(y1 + 0.55 * h)
    tx1, tx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    crop = frame[max(ty1, 0):ty2, max(tx1, 0):tx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)
    pixels = crop[~grass]
    if len(pixels) < 20:
        return None
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)


df = pd.read_csv(TRACKS_PATH)
by_frame = {f: g for f, g in df.groupby("frame")}

samples = []      # (track_id, L, a, b)
box_crops = {}    # track_id -> one full-body crop for the montage

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % SAMPLE_EVERY == 0 and frame_idx in by_frame:
        for row in by_frame[frame_idx].itertuples():
            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            color = torso_color(frame, x1, y1, x2, y2)
            if color is None:
                continue
            samples.append((row.track_id, *color))
            if row.track_id not in box_crops:
                body = frame[max(y1, 0):y2, max(x1, 0):x2]
                if body.size > 0:
                    box_crops[row.track_id] = cv2.resize(body, (50, 80))
    frame_idx += 1
cap.release()

feats = pd.DataFrame(samples, columns=["track_id", "L", "a", "b"])
counts = feats.groupby("track_id").size()
track_color = feats.groupby("track_id")[["L", "a", "b"]].median()
track_color = track_color[counts >= MIN_SAMPLES]

km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
track_color["cluster"] = km.fit_predict(track_color[["L", "a", "b"]])

out = df.merge(track_color[["cluster"]], left_on="track_id", right_index=True, how="left")
out["cluster"] = out["cluster"].fillna(-1).astype(int)
out.to_csv(OUT_PATH, index=False)

print("Tracks clustered:", len(track_color), "of", df["track_id"].nunique())
print("Rows without a cluster:", int((out["cluster"] == -1).sum()))
print()
for c in range(N_CLUSTERS):
    ids = track_color.index[track_color["cluster"] == c]
    rows = int((out["cluster"] == c).sum())
    lab = np.uint8([[km.cluster_centers_[c]]])
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0][0]
    print(f"Cluster {c}: {len(ids)} tracks, {rows} rows, avg color RGB=({r},{g},{b})")

# Montage: one row per cluster, up to 10 sample players each
cell_w, cell_h, margin, per_row = 50, 80, 70, 10
canvas = np.zeros((N_CLUSTERS * cell_h, margin + per_row * cell_w, 3), dtype=np.uint8)
for c in range(N_CLUSTERS):
    cv2.putText(canvas, f"C{c}", (10, c * cell_h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ids = [i for i in track_color.index[track_color["cluster"] == c] if i in box_crops]
    for k, tid in enumerate(ids[:per_row]):
        x0 = margin + k * cell_w
        canvas[c * cell_h:(c + 1) * cell_h, x0:x0 + cell_w] = box_crops[tid]
cv2.imwrite(MONTAGE_PATH, canvas)
print("\nSaved:", OUT_PATH)
print("Saved:", MONTAGE_PATH)