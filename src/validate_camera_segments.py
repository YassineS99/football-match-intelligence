
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

TRANSFORMS = Path("data/output/camera_transforms_hybrid.npy")
QUALITY = Path("data/output/camera_quality_hybrid.csv")
OUTPUT = Path("data/output/camera_segment_validation.csv")

Hs = np.load(TRANSFORMS)
df = pd.read_csv(QUALITY).sort_values("frame")

if len(Hs) != len(df):
    raise RuntimeError("Transform and quality row counts differ")

if df["frame"].tolist() != list(range(len(df))):
    raise RuntimeError("Frame indices are not consecutive")

VIDEO = "data/input/sample.mp4"
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

center = np.array([w / 2, h / 2, 1.0])

segment_id = 0
segment_start = 0
cumulative = np.eye(3)
records = []

for row in df.itertuples():
    f = int(row.frame)

    if f == 0:
        cumulative = np.eye(3)

    elif not bool(row.valid):
        # Failed transition into frame f:
        # start a new reference at frame f.
        segment_id += 1
        segment_start = f
        cumulative = np.eye(3)

    else:
        H = Hs[f]

        if not np.isfinite(H).all():
            raise RuntimeError(
                f"Valid frame {f} has non-finite transform"
            )

        cumulative = cumulative @ H

    mapped = cumulative @ center

    if abs(mapped[2]) < 1e-10:
        raise RuntimeError(f"Invalid mapping at frame {f}")

    mapped = mapped[:2] / mapped[2]

    scale = float(
        np.hypot(cumulative[0, 0], cumulative[1, 0])
    )

    records.append({
        "frame": f,
        "segment": segment_id,
        "segment_start": segment_start,
        "scale_vs_segment_start": scale,
        "center_x": float(mapped[0]),
        "center_y": float(mapped[1]),
        "center_displacement_px": float(
            np.linalg.norm(mapped - center[:2])
        ),
    })

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False)

print("LONGEST CAMERA SEGMENTS")
print("-" * 80)
print(
    "segment | frames | start-end | final scale | "
    "max displacement | scale range"
)

groups = sorted(
    result.groupby("segment"),
    key=lambda item: len(item[1]),
    reverse=True
)

for sid, group in groups[:10]:
    start = int(group["frame"].min())
    end = int(group["frame"].max())

    print(
        f"{sid:7d} | "
        f"{len(group):6d} | "
        f"{start:4d}-{end:<4d} | "
        f"{group['scale_vs_segment_start'].iloc[-1]:11.3f} | "
        f"{group['center_displacement_px'].max():16.1f} | "
        f"{group['scale_vs_segment_start'].min():.3f}-"
        f"{group['scale_vs_segment_start'].max():.3f}"
    )

print("\nSaved:", OUTPUT)
print("Note: These measurements do not prove alignment accuracy.")
