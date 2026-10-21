
import cv2
import numpy as np

VIDEO_PATH = "data/input/sample.mp4"

# Print a summary every 25 frames (1 second at 25 FPS).
REPORT_EVERY = 25

def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Approximate grass-color mask. This is a diagnostic,
    # not a complete pitch segmentation model.
    mask = (
        (hsv[:, :, 0] >= 30)
        & (hsv[:, :, 0] <= 90)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 35)
    ).astype(np.uint8) * 255

    return cv2.erode(mask, np.ones((5, 5), np.uint8))


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ok, prev_frame = cap.read()

if not ok:
    raise RuntimeError("Could not read the first video frame.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_mask = grass_mask(prev_frame)

frame_idx = 0
results = []

print(
    "frame | scale | translation px | tracked | inliers | "
    "inlier ratio | status"
)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=600,
        qualityLevel=0.005,
        minDistance=8,
        mask=prev_mask
    )

    scale = float("nan")
    translation = float("nan")
    tracked = 0
    inliers = 0
    ratio = 0.0
    status = "LOW FEATURES"

    if pts is not None and len(pts) >= 10:
        nxt, flow_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None
        )

        if nxt is not None and flow_status is not None:
            good = flow_status.ravel() == 1
            old_pts = pts.reshape(-1, 2)[good]
            new_pts = nxt.reshape(-1, 2)[good]

            tracked = len(old_pts)

            if tracked >= 10:
                M, inlier_mask = cv2.estimateAffinePartial2D(
                    new_pts,
                    old_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0
                )

                if M is not None and inlier_mask is not None:
                    inliers = int(inlier_mask.sum())
                    ratio = inliers / tracked

                    scale = float(
                        np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                    )

                    translation = float(
                        np.linalg.norm(M[:, 2])
                    )

                    if ratio >= 0.5 and inliers >= 30:
                        status = "OK"
                    else:
                        status = "WEAK TRACKING"

                    if scale < 0.90 or scale > 1.10:
                        status += " | LARGE SCALE CHANGE"

                    if translation > 30:
                        status += " | LARGE MOVEMENT"

                else:
                    status = "TRANSFORM FAILED"

    results.append(
        (frame_idx, scale, translation, tracked, inliers, ratio, status)
    )

    if frame_idx % REPORT_EVERY == 0 or status != "OK":
        print(
            f"{frame_idx:5d} | "
            f"{scale:5.3f} | "
            f"{translation:14.2f} | "
            f"{tracked:7d} | "
            f"{inliers:7d} | "
            f"{ratio:11.2f} | {status}"
        )

    prev_gray = gray
    prev_mask = grass_mask(frame)

cap.release()

print("\nDiagnostic complete.")
print(f"Frames analyzed: {frame_idx}")
print("No existing tracking or homography files were modified.")
