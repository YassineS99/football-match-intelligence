
import pandas as pd
from pathlib import Path

INPUT = Path("data/output/camera_quality_hybrid.csv")

df = pd.read_csv(INPUT).sort_values("frame").reset_index(drop=True)

# Frame 0 is the reference, not an estimated transition.
transitions = df[df["frame"] > 0].copy()

failed = transitions[transitions["method"] == "failed"]
orb = transitions[transitions["method"] == "orb"]

print("CAMERA QUALITY SUMMARY")
print("-" * 50)

print(f"Total transitions: {len(transitions)}")
print(f"Failed transitions: {len(failed)}")
print(f"ORB fallback transitions: {len(orb)}")

print("\nFAILED FRAME RANGES")
print("-" * 50)

def contiguous_ranges(numbers):
    numbers = sorted(set(int(n) for n in numbers))

    if not numbers:
        return []

    ranges = []
    start = previous = numbers[0]

    for n in numbers[1:]:
        if n == previous + 1:
            previous = n
        else:
            ranges.append((start, previous))
            start = previous = n

    ranges.append((start, previous))
    return ranges


for start, end in contiguous_ranges(failed["frame"]):
    print(f"Frames {start:4d}-{end:4d} ({end-start+1} failures)")


print("\nCONTINUOUS VALID SEGMENTS")
print("-" * 50)

# A failed transition into frame f breaks the chain
# between frame f-1 and frame f.
segments = []
segment_start = 0
previous_frame = 0

for row in transitions.itertuples():
    f = int(row.frame)

    if f != previous_frame + 1:
        raise RuntimeError("Missing or duplicate frame indices")

    if row.method == "failed":
        segments.append((segment_start, f - 1))
        segment_start = f

    previous_frame = f

segments.append((segment_start, previous_frame))

segments = sorted(
    segments,
    key=lambda s: s[1] - s[0],
    reverse=True
)

for start, end in segments[:15]:
    count = end - start + 1
    duration = count / 25.0

    print(
        f"Frames {start:4d}-{end:4d} | "
        f"{count:4d} frames | "
        f"{duration:5.2f} sec"
    )


print("\nORB FALLBACK RANGES")
print("-" * 50)

for start, end in contiguous_ranges(orb["frame"]):
    print(f"Frames {start:4d}-{end:4d}")


print("\nANALYSIS COMPLETE")
print("These are connectivity segments, not validated pitch mappings.")
