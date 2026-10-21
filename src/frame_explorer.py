"""Read and annotate individual source frames, without running detectors or writing files."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROLE_RGB = {
    "team_a_outfield": (139, 30, 63),
    "team_b_outfield": (40, 120, 184),
    "referee_candidate": (227, 173, 27),
    "goalkeeper_candidate": (46, 139, 87),
    "unknown": (127, 135, 144),
}


def read_source_frame(path, frame_number):
    """Return the requested decoded source frame as RGB pixels (read-only I/O)."""
    path = Path(path).expanduser()
    try:
        frame_number = int(frame_number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Frame number must be an integer.") from exc
    if frame_number < 0:
        raise ValueError("Frame number must be nonnegative.")
    if not path.is_file():
        raise FileNotFoundError(f"Source video is not available: {path}")

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        reader.release()
        raise ValueError(f"Cannot open source video: {path}")
    try:
        if not reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number) and frame_number != 0:
            raise ValueError(f"Cannot seek to source frame {frame_number}.")
        success, bgr = reader.read()
        if not success or bgr is None:
            raise ValueError(f"Cannot decode source frame {frame_number}.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        reader.release()


def _finite_xy(record, x_key, y_key):
    try:
        x = float(record[x_key])
        y = float(record[y_key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def draw_frame_annotations(image_rgb, rows, *, boxes=True, foot_points=True):
    """Draw *provisional* track IDs on a fresh copy of a decoded RGB frame.

    The original tracking frame, CSV and source video are never modified.
    Bounding boxes are optional, because older tracking CSVs may have only
    foot_x/foot_y. Missing or nonfinite geometry is skipped, not fabricated.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 video frame.")
    result = image.copy()
    height, width = result.shape[:2]
    if not height or not width:
        return result
    if rows is None or rows.empty:
        return result

    for _, observation in rows.iterrows():
        role = str(observation.get("predicted_role", "unknown"))
        if role not in ROLE_RGB:
            role = "unknown"
        color = ROLE_RGB[role]
        track_id = str(observation.get("track_id", "?"))
        if boxes:
            top_left = _finite_xy(observation, "x1", "y1")
            bottom_right = _finite_xy(observation, "x2", "y2")
            if top_left and bottom_right:
                x1, y1 = top_left
                x2, y2 = bottom_right
                # Invalid or fully out-of-view boxes are ignored.
                if x2 > x1 and y2 > y1 and x2 >= 0 and y2 >= 0 and x1 < width and y1 < height:
                    a = (min(max(int(round(x1)), 0), width - 1), min(max(int(round(y1)), 0), height - 1))
                    b = (min(max(int(round(x2)), 0), width - 1), min(max(int(round(y2)), 0), height - 1))
                    cv2.rectangle(result, a, b, color, 2)
                    cv2.putText(result, f"ID {track_id}", (a[0], max(a[1] - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)

        if foot_points:
            foot = _finite_xy(observation, "foot_x", "foot_y")
            if foot:
                x, y = foot
                if 0 <= x < width and 0 <= y < height:
                    point = (int(round(x)), int(round(y)))
                    cv2.circle(result, point, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(result, point, 7, (255, 255, 255), 1, cv2.LINE_AA)
    return result
