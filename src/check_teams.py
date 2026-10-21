import cv2
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
TRACKS_PATH = "data/output/tracks_teams.csv"
OUT_PATH = "data/output/team_check.jpg"
FRAMES = [100, 500, 900]

# cluster id -> (label, BGR color)
LABELS = {
    0: ("Red team", (0, 0, 255)),
    1: ("Referee", (0, 255, 255)),
    2: ("Blue team", (255, 128, 0)),
    3: ("Other", (160, 160, 160)),
}

df = pd.read_csv(TRACKS_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
panels = []

for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    for r in df[df["frame"] == f].itertuples():
        name, color = LABELS.get(r.cluster, ("?", (255, 255, 255)))
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, name, (p1[0], p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    panels.append(cv2.resize(frame, (960, 540)))

cap.release()
cv2.imwrite(OUT_PATH, cv2.vconcat(panels))
print("Saved:", OUT_PATH)