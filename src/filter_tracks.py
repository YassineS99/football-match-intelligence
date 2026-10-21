import pandas as pd

IN_PATH = "data/output/tracks.csv"
OUT_PATH = "data/output/tracks_filtered.csv"
FPS = 25
MIN_FRAMES = FPS  # drop tracks shorter than 1 second

df = pd.read_csv(IN_PATH)
lengths = df.groupby("track_id")["frame"].nunique()
keep_ids = lengths[lengths >= MIN_FRAMES].index

filtered = df[df["track_id"].isin(keep_ids)]
filtered.to_csv(OUT_PATH, index=False)

per_frame = filtered.groupby("frame")["track_id"].nunique()

print("Rows before:", len(df))
print("Rows after:", len(filtered), f"({len(filtered) / len(df):.0%} kept)")
print("Track IDs before:", df["track_id"].nunique())
print("Track IDs after:", filtered["track_id"].nunique())
print("Players per frame (mean) after:", round(per_frame.mean(), 1))
print("Frames with no tracked players:", df["frame"].nunique() - per_frame.shape[0])