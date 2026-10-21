import pandas as pd

TRACKS_PATH = "data/output/tracks_roles.csv"
KEEPER_IDS = [11, 1607, 792, 948, 1291]

df = pd.read_csv(TRACKS_PATH)
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = df["y2"]  # feet position

print("Keeper vs team average distance (pixels) per track, in frames where both exist:\n")
for tid in KEEPER_IDS:
    k = df[df["track_id"] == tid]
    frames = k["frame"].unique()
    d = {}
    for team in ["red", "blue"]:
        t = df[(df["role"] == team) & (df["frame"].isin(frames))]
        mean_pos = t.groupby("frame")[["cx", "cy"]].mean()
        merged = k.set_index("frame")[["cx", "cy"]].join(mean_pos, rsuffix="_team", how="inner")
        d[team] = (((merged["cx"] - merged["cx_team"]) ** 2 +
                    (merged["cy"] - merged["cy_team"]) ** 2) ** 0.5).mean()
    print(f"track {tid}: frames={len(frames)}, dist_to_red={d['red']:.0f}, dist_to_blue={d['blue']:.0f}")