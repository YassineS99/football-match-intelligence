import cv2
import pandas as pd

VIDEO_PATH = "data/input/sample.mp4"
IN_PATH = "data/output/tracks_teams.csv"
OUT_PATH = "data/output/tracks_roles.csv"
CHECK_PATH = "data/output/roles_check.jpg"
FRAMES = [100, 500, 900]

KEEPER_IDS = [11, 1607, 792, 948, 1291]
REFEREE_IDS = [797, 1391, 963, 689, 787, 1215]

CLUSTER_ROLE = {0: "red", 1: "referee", 2: "blue", 3: "other"}
COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 128, 0),
    "keeper": (0, 255, 0),
    "referee": (0, 255, 255),
    "other": (160, 160, 160),
}

df = pd.read_csv(IN_PATH)
df["role"] = df["cluster"].map(CLUSTER_ROLE).fillna("other")
df.loc[df["track_id"].isin(KEEPER_IDS), "role"] = "keeper"
df.loc[df["track_id"].isin(REFEREE_IDS), "role"] = "referee"
df.to_csv(OUT_PATH, index=False)

print("Role summary:")
summary = df.groupby("role").agg(tracks=("track_id", "nunique"), rows=("frame", "size"))
print(summary)

cap = cv2.VideoCapture(VIDEO_PATH)
panels = []
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    for r in df[df["frame"] == f].itertuples():
        color = COLORS[r.role]
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, r.role, (p1[0], p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    panels.append(cv2.resize(frame, (960, 540)))
cap.release()
cv2.imwrite(CHECK_PATH, cv2.vconcat(panels))
print("Saved:", OUT_PATH)
print("Saved:", CHECK_PATH)