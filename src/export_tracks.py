import pandas as pd
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = "data/input/sample.mp4"
CSV_PATH = "data/output/tracks.csv"
PERSON_CLASS_ID = 0

model = YOLO("yolov8x.pt")
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
frames = sv.get_video_frames_generator(VIDEO_PATH)

rows = []
for frame_idx, frame in enumerate(tqdm(frames, total=video_info.total_frames, desc="Tracking")):
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS_ID],
        conf=0.25,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)

    if detections.tracker_id is None:
        continue

    for box, conf, tid in zip(detections.xyxy, detections.confidence, detections.tracker_id):
        x1, y1, x2, y2 = box
        rows.append({
            "frame": frame_idx,
            "track_id": int(tid),
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "conf": float(conf),
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

# --- ID stability stats ---
lengths = df.groupby("track_id")["frame"].nunique()
fps = video_info.fps

print("\nSaved:", CSV_PATH)
print("Total detections rows:", len(df))
print("Unique track IDs:", len(lengths))
print("Players per frame (mean):", round(df.groupby("frame")["track_id"].nunique().mean(), 1))
print(f"IDs lasting < 1 second: {(lengths < fps).sum()}")
print(f"IDs lasting 1-5 seconds: {((lengths >= fps) & (lengths < 5 * fps)).sum()}")
print(f"IDs lasting >= 5 seconds: {(lengths >= 5 * fps).sum()}")
print("Longest track (frames):", lengths.max())
print("Median track length (frames):", int(lengths.median()))