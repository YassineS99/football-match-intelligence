import argparse
import csv
import json
import math
import os
import sys
import tempfile
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

from detect_objects import resolve_device, seed_generators


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "input" / "sample.mp4"
DEFAULT_MODEL = ROOT / "yolov8x.pt"
DEFAULT_DETECTIONS = ROOT / "data" / "output" / "detection" / "detections.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "tracking"
DEFAULT_MIN_CONFIRMED_FRAMES = 10
RANDOM_SEED = 42
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
TRACK_CSV_FIELDS = [
    "run_id",
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "foot_x",
    "foot_y",
]
DETECTION_FIELDS = {
    "frame_number",
    "object_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
}
OUTPUT_FILENAMES = (
    "annotated_tracking.mp4",
    "tracks.csv",
    "tracks_confirmed.csv",
    "tracking_summary.json",
)
TRACKING_CAVEAT = (
    "Track IDs are tracker assignments, not guaranteed real-world player "
    "identities; IDs may switch or fragment after occlusions and are not "
    "guaranteed consistent across environments."
)
CAMERA_COMPENSATION_NOTE = (
    "No camera compensation is applied. The current camera quality data has "
    "60 failed transitions; any future compensation must not silently cross "
    "those failures."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Track person detections with Ultralytics ByteTrack. IDs are "
            "provisional and may switch after occlusions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local Ultralytics YOLO weights used when --detections is omitted.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help=(
            "Optional detections.csv from src/detect_objects.py. Its sibling "
            "detection_summary.json is required to verify frame coverage. "
            "Sports balls are retained as untracked detections."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or CUDA index (default: auto).",
    )
    parser.add_argument(
        "--detection-conf",
        "--conf",
        dest="detection_conf",
        type=float,
        default=None,
        help=(
            "Minimum confidence passed from YOLO to tracking. Defaults to "
            "ByteTrack's configured low threshold so low-score association "
            "detections are available. --conf is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=None,
        help="Override ByteTrack's low-score association threshold from YAML.",
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=None,
        help="Override ByteTrack's first-stage association threshold from YAML.",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=None,
        help="Override ByteTrack's new-track threshold from YAML.",
    )
    parser.add_argument(
        "--min-confirmed-frames",
        type=int,
        default=DEFAULT_MIN_CONFIRMED_FRAMES,
        help=(
            "Minimum distinct frames with an assigned ID to call a person "
            f"a confirmed candidate (default: {DEFAULT_MIN_CONFIRMED_FRAMES})."
        ),
    )
    parser.add_argument(
        "--confirmed-only",
        action="store_true",
        help=(
            "Show only confirmed person candidates in the annotated video. "
            "Raw detections and provisional tracks remain in CSV outputs."
        ),
    )
    parser.add_argument(
        "--display-conf",
        type=float,
        default=0.0,
        help=(
            "Minimum confidence for drawing boxes in the video only; does not "
            "filter tracker inputs or CSV records (default: 0)."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def validate_args(args, parser=None):
    if (
        args.detection_conf is not None
        and (
            not math.isfinite(args.detection_conf)
            or not 0.0 < args.detection_conf <= 1.0
        )
    ):
        message = "--detection-conf must be greater than 0 and at most 1."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not math.isfinite(args.display_conf) or not 0.0 <= args.display_conf <= 1.0:
        message = "--display-conf must be between 0 and 1."
        if parser:
            parser.error(message)
        raise ValueError(message)
    for option_name in (
        "track_low_thresh",
        "track_high_thresh",
        "new_track_thresh",
    ):
        value = getattr(args, option_name)
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            message = f"--{option_name.replace('_', '-')} must be between 0 and 1."
            if parser:
                parser.error(message)
            raise ValueError(message)
    if args.min_confirmed_frames <= 0:
        message = "--min-confirmed-frames must be a positive integer."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if args.start_frame < 0:
        message = "--start-frame must be a non-negative integer."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if args.max_frames is not None and args.max_frames <= 0:
        message = "--max-frames must be a positive integer when specified."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not str(args.device).strip():
        message = "--device cannot be empty."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not args.output_dir:
        message = "--output-dir cannot be empty."
        if parser:
            parser.error(message)
        raise ValueError(message)


def load_tracker_config(overrides=None):
    import ultralytics
    import yaml

    config_path = (
        Path(ultralytics.__file__).resolve().parent
        / "cfg"
        / "trackers"
        / "bytetrack.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics ByteTrack configuration not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config.get("tracker_type") != "bytetrack":
        raise ValueError(f"Unexpected tracker type in {config_path}.")
    required = {
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
    }
    if not required.issubset(config):
        raise ValueError(f"Incomplete ByteTrack configuration: {config_path}")
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    if config["track_low_thresh"] > config["track_high_thresh"]:
        raise ValueError(
            "ByteTrack track_low_thresh cannot exceed track_high_thresh."
        )
    return SimpleNamespace(**config)


def create_tracker(tracker_config=None):
    from ultralytics.trackers.byte_tracker import BYTETracker

    return BYTETracker(tracker_config or load_tracker_config())


def resolve_detection_confidence(requested, tracker_config):
    return float(
        tracker_config.track_low_thresh
        if requested is None
        else requested
    )


def validate_precomputed_confidence(source_confidence, requested_confidence):
    if source_confidence is None:
        return
    if (
        not math.isfinite(source_confidence)
        or not 0.0 < source_confidence <= 1.0
    ):
        raise ValueError("Detection summary has an invalid confidence threshold.")
    if source_confidence > requested_confidence:
        raise ValueError(
            "The precomputed detections were generated at confidence "
            f"{source_confidence:.3f}, above requested "
            f"{requested_confidence:.3f}; lower-confidence boxes are missing. "
            "Run fresh YOLO inference with --detection-conf at or below that "
            "threshold."
        )


def select_visualization_rows(
    rows, confirmed_track_ids, confirmed_only, display_confidence
):
    selected = []
    for row in rows:
        if float(row["confidence"]) < display_confidence:
            continue
        if int(row["class_id"]) == SPORTS_BALL_CLASS_ID:
            selected.append(row)
        elif not confirmed_only:
            selected.append(row)
        elif (
            row["track_id"] != ""
            and int(row["track_id"]) in confirmed_track_ids
        ):
            selected.append(row)
    return selected


def create_tracking_run_id():
    return uuid.uuid4().hex


def parse_track_csv_row(row):
    parsed = {
        "run_id": row["run_id"],
        "frame": int(row["frame"]),
        "class_id": int(row["class_id"]),
    }
    parsed["track_id"] = row["track_id"]
    for key in (
        "timestamp",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    ):
        parsed[key] = float(row[key])
    for key in ("foot_x", "foot_y"):
        parsed[key] = float(row[key]) if row[key] else ""
    return parsed


def write_confirmed_tracks(raw_csv_path, confirmed_csv_path, confirmed_ids):
    count = 0
    with raw_csv_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        with confirmed_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(
                destination, fieldnames=TRACK_CSV_FIELDS
            )
            writer.writeheader()
            for row in reader:
                if (
                    int(row["class_id"]) == PERSON_CLASS_ID
                    and row["track_id"]
                    and int(row["track_id"]) in confirmed_ids
                ):
                    writer.writerow(row)
                    count += 1
    return count


def write_annotated_video(
    input_path,
    output_path,
    tracks_csv_path,
    start_frame,
    end_frame,
    width,
    height,
    fps,
    confirmed_ids,
    confirmed_only,
    display_confidence,
):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen input video for annotation: {input_path}")
    writer = None
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(
                f"Could not seek to frame {start_frame} for annotation."
            )
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize MP4 writer at {output_path}.")

        trails = defaultdict(lambda: deque(maxlen=30))
        with tracks_csv_path.open(
            "r", newline="", encoding="utf-8"
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            pending = next(reader, None)
            progress = tqdm(
                total=end_frame - start_frame,
                desc="Rendering",
                unit="frame",
                file=sys.stderr,
            )
            try:
                for frame_number in range(start_frame, end_frame):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Could not reread video frame {frame_number}."
                        )
                    frame_rows = []
                    while (
                        pending is not None
                        and int(pending["frame"]) < frame_number
                    ):
                        raise ValueError(
                            "Tracking CSV frame order does not match source video."
                        )
                    while (
                        pending is not None
                        and int(pending["frame"]) == frame_number
                    ):
                        frame_rows.append(parse_track_csv_row(pending))
                        pending = next(reader, None)
                    visible_rows = select_visualization_rows(
                        frame_rows,
                        confirmed_ids,
                        confirmed_only,
                        display_confidence,
                    )
                    writer.write(draw_tracking(frame, visible_rows, trails))
                    progress.update(1)
            finally:
                progress.close()
            if pending is not None:
                raise ValueError(
                    "Tracking CSV contains records beyond the rendered frame range."
                )
    finally:
        if writer is not None:
            writer.release()
        capture.release()


@dataclass
class TrackingSession:
    tracker: object
    tracker_config: object
    last_frame: int | None = None
    id_offset: int = 0
    max_local_id: int = 0

    @classmethod
    def create(cls, tracker_config=None):
        tracker_config = tracker_config or load_tracker_config()
        return cls(
            tracker=create_tracker(tracker_config),
            tracker_config=tracker_config,
        )

    def reset(self):
        self.id_offset += self.max_local_id
        self.tracker = create_tracker(self.tracker_config)
        self.last_frame = None
        self.max_local_id = 0


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def update_tracking_frame(session, detections, frame_number, fps, image_shape):
    import torch
    from ultralytics.engine.results import Boxes

    if session.last_frame is not None and frame_number != session.last_frame + 1:
        session.reset()

    people = [
        detection
        for detection in detections
        if int(detection["class_id"]) == PERSON_CLASS_ID
    ]
    person_boxes = np.asarray(
        [
            [
                float(item["x1"]),
                float(item["y1"]),
                float(item["x2"]),
                float(item["y2"]),
                float(item["confidence"]),
                float(item["class_id"]),
            ]
            for item in people
        ],
        dtype=np.float32,
    ).reshape(-1, 6)
    height, width = image_shape[:2]
    boxes = Boxes(torch.from_numpy(person_boxes), (height, width))
    tracked = session.tracker.update(boxes)

    assigned_ids = {}
    if tracked.size:
        if tracked.ndim != 2 or tracked.shape[1] < 8:
            raise RuntimeError(
                "ByteTrack returned an unexpected result shape; expected "
                "[x1, y1, x2, y2, track_id, confidence, class_id, detection_index]."
            )
        for result in tracked:
            detection_index = int(result[7])
            local_track_id = int(result[4])
            assigned_ids[detection_index] = session.id_offset + local_track_id
            session.max_local_id = max(session.max_local_id, local_track_id)

    rows = []
    person_index = 0
    for detection in detections:
        class_id = int(detection["class_id"])
        if class_id not in {PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID}:
            continue
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        center_x = (x1 + x2) / 2.0
        if class_id == PERSON_CLASS_ID:
            track_id = assigned_ids.get(person_index, "")
            person_index += 1
        else:
            track_id = ""
        rows.append({
            "frame": int(frame_number),
            "timestamp": float(frame_number / fps),
            "track_id": track_id,
            "class_id": class_id,
            "confidence": float(detection["confidence"]),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": (y1 + y2) / 2.0,
            "foot_x": center_x if class_id == PERSON_CLASS_ID else "",
            "foot_y": y2 if class_id == PERSON_CLASS_ID else "",
        })

    session.last_frame = int(frame_number)
    return rows


def records_from_result(result, class_names, confidence_threshold):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _as_numpy(boxes.xyxy)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    class_ids = _as_numpy(boxes.cls).reshape(-1).astype(int)
    if not (len(xyxy) == len(confidences) == len(class_ids)):
        raise ValueError("Model returned mismatched box, confidence, and class arrays.")

    records = []
    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        if (
            int(class_id) not in {PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID}
            or float(confidence) < confidence_threshold
        ):
            continue
        records.append({
            "class_id": int(class_id),
            "object_class": class_names.get(int(class_id), str(class_id)),
            "confidence": float(confidence),
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        })
    return records


def _read_detection_metadata(detections_path, input_path):
    metadata_path = detections_path.with_name("detection_summary.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Detection metadata not found: {metadata_path}. The detection "
            "CSV needs its companion summary to verify frame coverage."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    recorded_input = metadata.get("input_video")
    if not recorded_input:
        raise ValueError(f"Missing input_video in {metadata_path}.")
    if Path(recorded_input).expanduser().resolve() != input_path.expanduser().resolve():
        raise ValueError(
            "Detection CSV metadata refers to a different input video: "
            f"{recorded_input}"
        )
    try:
        start = int(metadata["start_frame"])
        end_exclusive = int(metadata["end_frame_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Detection summary has no valid processed frame range: {metadata_path}"
        ) from exc
    if start < 0 or end_exclusive <= start:
        raise ValueError(f"Invalid processed frame range in {metadata_path}.")

    class_names = {
        int(class_id): str(name).strip().lower()
        for class_id, name in metadata.get("model_classes", {}).items()
    }
    if (
        class_names.get(PERSON_CLASS_ID) != "person"
        or class_names.get(SPORTS_BALL_CLASS_ID) != "sports ball"
    ):
        raise ValueError(
            f"{metadata_path} must identify COCO person and sports ball classes."
        )
    return metadata, start, end_exclusive, class_names


def validate_detection_csv(detections_path):
    required = DETECTION_FIELDS
    previous_frame = -1
    last_frame = -1
    with detections_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{detections_path} must contain columns: "
                + ", ".join(sorted(required))
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame_number"])
                confidence = float(row["confidence"])
                box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid detection values at CSV row {row_number}."
                ) from exc
            if frame < previous_frame:
                raise ValueError(
                    f"Detection CSV is not ordered by frame at row {row_number}."
                )
            if (
                not math.isfinite(confidence)
                or not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError(
                    f"Invalid confidence or bounding box at CSV row {row_number}."
                )
            if row["object_class"].strip().lower() not in {
                "person",
                "sports ball",
            }:
                raise ValueError(
                    f"Unsupported class at CSV row {row_number}: "
                    f"{row['object_class']!r}."
                )
            previous_frame = frame
            last_frame = frame
    return last_frame


def iter_detection_frames(detections_path, start_frame, end_frame, class_names, confidence):
    with detections_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        pending = next(reader, None)
        for frame_number in range(start_frame, end_frame):
            frame_detections = []
            while pending is not None and int(pending["frame_number"]) < frame_number:
                pending = next(reader, None)
            while (
                pending is not None
                and int(pending["frame_number"]) == frame_number
            ):
                class_name = pending["object_class"].strip().lower()
                class_id = next(
                    (
                        key
                        for key, value in class_names.items()
                        if value == class_name
                    ),
                    None,
                )
                row_confidence = float(pending["confidence"])
                if class_id is not None and row_confidence >= confidence:
                    frame_detections.append({
                        "class_id": class_id,
                        "object_class": class_name,
                        "confidence": row_confidence,
                        "x1": float(pending["x1"]),
                        "y1": float(pending["y1"]),
                        "x2": float(pending["x2"]),
                        "y2": float(pending["y2"]),
                    })
                pending = next(reader, None)
            yield frame_detections


def create_temp_paths(output_dir):
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite tracking outputs: "
            + ", ".join(existing)
            + ". Choose another --output-dir."
        )

    temporary = {}
    try:
        for key, suffix in (
            ("video", ".mp4"),
            ("csv", ".csv"),
            ("confirmed_csv", ".csv"),
            ("summary", ".json"),
        ):
            descriptor, name = tempfile.mkstemp(
                prefix=f".tracking_{key}_", suffix=suffix, dir=output_dir
            )
            os.close(descriptor)
            temporary[key] = Path(name)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return paths, temporary


def draw_tracking(frame, rows, trails):
    annotated = frame.copy()
    height, width = frame.shape[:2]
    for row in rows:
        x1 = int(np.clip(round(row["x1"]), 0, width - 1))
        y1 = int(np.clip(round(row["y1"]), 0, height - 1))
        x2 = int(np.clip(round(row["x2"]), 0, width - 1))
        y2 = int(np.clip(round(row["y2"]), 0, height - 1))
        class_id = int(row["class_id"])
        track_id = row["track_id"]
        if class_id == PERSON_CLASS_ID:
            color = (0, 220, 0)
            label = (
                f"person ID {track_id}"
                if track_id != ""
                else "person ID pending"
            )
            if track_id != "":
                foot = (round(row["foot_x"]), round(row["foot_y"]))
                trails[int(track_id)].append(foot)
                points = list(trails[int(track_id)])
                if len(points) > 1:
                    cv2.polylines(
                        annotated,
                        [np.asarray(points, dtype=np.int32)],
                        False,
                        color,
                        2,
                    )
        else:
            color = (0, 165, 255)
            label = "sports ball (untracked)"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} {row['confidence']:.2f}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


class TrackStatistics:
    def __init__(self):
        self.track_frames = defaultdict(set)
        self.tracked_detections = 0
        self.unassigned_people = 0
        self.ball_detections = 0

    def add(self, row):
        class_id = int(row["class_id"])
        if class_id == SPORTS_BALL_CLASS_ID:
            self.ball_detections += 1
        elif row["track_id"] == "" or row["track_id"] is None:
            self.unassigned_people += 1
        else:
            self.tracked_detections += 1
            self.track_frames[int(row["track_id"])].add(int(row["frame"]))

    def summary(self):
        durations = {
            track_id: len(frame_numbers)
            for track_id, frame_numbers in self.track_frames.items()
        }
        fragments = {}
        for track_id, frame_numbers in self.track_frames.items():
            ordered = sorted(frame_numbers)
            fragments[track_id] = 1 + sum(
                current != previous + 1
                for previous, current in zip(ordered, ordered[1:])
            )
        duration_values = list(durations.values())
        distribution = {
            "min_frames": int(np.min(duration_values)) if duration_values else 0,
            "median_frames": (
                float(np.median(duration_values)) if duration_values else 0.0
            ),
            "mean_frames": (
                float(np.mean(duration_values)) if duration_values else 0.0
            ),
            "p90_frames": (
                float(np.percentile(duration_values, 90))
                if duration_values else 0.0
            ),
            "max_frames": int(np.max(duration_values)) if duration_values else 0,
        }
        return {
            "total_tracked_detections": self.tracked_detections,
            "person_detections_with_track_id": self.tracked_detections,
            "person_detections_without_track_id": self.unassigned_people,
            "sports_ball_detections_untracked": self.ball_detections,
            "unique_track_ids": len(durations),
            "tracks_lasting_at_least_5_frames": sum(
                duration >= 5 for duration in duration_values
            ),
            "tracks_lasting_at_least_10_frames": sum(
                duration >= 10 for duration in duration_values
            ),
            "tracks_lasting_at_least_25_frames": sum(
                duration >= 25 for duration in duration_values
            ),
            "track_duration_distribution": distribution,
            "tracks_with_multiple_fragments": sum(
                count > 1 for count in fragments.values()
            ),
            "total_extra_track_fragments": sum(
                max(0, count - 1) for count in fragments.values()
            ),
        }

    def confirmed_track_ids(self, minimum_frames):
        return {
            track_id
            for track_id, frame_numbers in self.track_frames.items()
            if len(frame_numbers) >= minimum_frames
        }


def count_failed_camera_transitions(path):
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"frame", "method", "valid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            return None
        count = 0
        for row in reader:
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError):
                return None
            if frame > 0 and (
                row["method"].strip().lower() == "failed"
                or row["valid"].strip().lower() in {"false", "0"}
            ):
                count += 1
        return count


def run_tracking(args):
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    detections_path = (
        args.detections.expanduser().resolve()
        if args.detections is not None
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if detections_path is None and not model_path.is_file():
        raise FileNotFoundError(f"YOLO model weights not found: {model_path}")
    if detections_path is not None and not detections_path.is_file():
        raise FileNotFoundError(f"Detections CSV not found: {detections_path}")

    detection_metadata = None
    detection_source_confidence = None
    class_names = {PERSON_CLASS_ID: "person", SPORTS_BALL_CLASS_ID: "sports ball"}
    if detections_path is not None:
        detection_metadata, covered_start, covered_end, class_names = (
            _read_detection_metadata(detections_path, input_path)
        )
        validate_detection_csv(detections_path)
        recorded_confidence = detection_metadata.get("confidence_threshold")
        if recorded_confidence is not None:
            detection_source_confidence = float(recorded_confidence)

    tracker_config = load_tracker_config({
        "track_low_thresh": args.track_low_thresh,
        "track_high_thresh": args.track_high_thresh,
        "new_track_thresh": args.new_track_thresh,
    })
    detection_confidence = resolve_detection_confidence(
        args.detection_conf, tracker_config
    )
    if detections_path is not None:
        validate_precomputed_confidence(
            detection_source_confidence, detection_confidence
        )
        if detection_source_confidence is None:
            print(
                "WARNING: precomputed detection confidence provenance is "
                "unknown; low-score ByteTrack association cannot be verified.",
                file=sys.stderr,
            )
        elif detection_source_confidence > tracker_config.track_low_thresh:
            print(
                "WARNING: the precomputed CSV cutoff is above ByteTrack's "
                "low association threshold; low-score association candidates "
                "were already discarded.",
                file=sys.stderr,
            )
    elif detection_confidence > tracker_config.track_low_thresh:
        print(
            "WARNING: --detection-conf is above ByteTrack's low association "
            "threshold; low-score candidates will not reach the tracker.",
            file=sys.stderr,
        )

    device = (
        resolve_device(args.device)
        if detections_path is None
        else "not used (precomputed detections)"
    )
    seed_generators(RANDOM_SEED)

    model = None
    if detections_path is None:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        normalized_names = {
            int(class_id): str(name).strip().lower()
            for class_id, name in model.names.items()
        }
        if normalized_names.get(PERSON_CLASS_ID) != "person":
            raise ValueError("YOLO model must include COCO person class ID 0.")
        class_names = {
            class_id: class_name
            for class_id, class_name in normalized_names.items()
            if class_id in {PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID}
        }
        if class_names.get(SPORTS_BALL_CLASS_ID) != "sports ball":
            raise ValueError("YOLO model must include COCO sports ball class ID 32.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths, temporary = create_temp_paths(output_dir)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot open video: {input_path}")

    writer = None
    session = TrackingSession.create(tracker_config)
    statistics = TrackStatistics()
    run_id = create_tracking_run_id()
    processed = 0
    total_source_detections = 0
    start_frame = args.start_frame
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid frame count or dimensions.")
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Input video has invalid FPS: {fps}.")
        if start_frame >= frame_count:
            raise ValueError(
                f"--start-frame {start_frame} is outside the video; "
                f"valid frames are 0 through {frame_count - 1}."
            )
        remaining = frame_count - start_frame
        frame_limit = min(args.max_frames or remaining, remaining)
        end_frame = start_frame + frame_limit
        if detections_path is not None:
            if start_frame < covered_start or end_frame > covered_end:
                raise ValueError(
                    f"Detection CSV covers frames [{covered_start}, "
                    f"{covered_end}), but requested range is "
                    f"[{start_frame}, {end_frame}). Use fresh YOLO inference "
                    "or request a fully covered frame range."
                )

        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(f"Could not seek to frame {start_frame}.")
        detection_frames = None
        if detections_path is not None:
            detection_frames = iter_detection_frames(
                detections_path,
                start_frame,
                end_frame,
                class_names,
                detection_confidence,
            )

        with temporary["csv"].open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=TRACK_CSV_FIELDS
            )
            csv_writer.writeheader()
            progress = tqdm(
                total=frame_limit,
                desc="Tracking",
                unit="frame",
                file=sys.stderr,
            )
            try:
                for frame_number in range(start_frame, end_frame):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Could not read expected video frame {frame_number}."
                        )
                    if detection_frames is not None:
                        detections = next(detection_frames)
                    else:
                        results = model.predict(
                            source=frame,
                            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
                            conf=detection_confidence,
                            device=device,
                            verbose=False,
                        )
                        detections = records_from_result(
                            results[0], class_names, detection_confidence
                        )

                    total_source_detections += len(detections)
                    frame_rows = update_tracking_frame(
                        session, detections, frame_number, fps, frame.shape
                    )
                    for row in frame_rows:
                        row["run_id"] = run_id
                        statistics.add(row)
                        csv_writer.writerow(row)
                    processed += 1
                    progress.update(1)
            finally:
                progress.close()

        capture.release()
        capture = None

        confirmed_track_ids = statistics.confirmed_track_ids(
            args.min_confirmed_frames
        )
        confirmed_detection_count = write_confirmed_tracks(
            temporary["csv"],
            temporary["confirmed_csv"],
            confirmed_track_ids,
        )
        write_annotated_video(
            input_path,
            temporary["video"],
            temporary["csv"],
            start_frame,
            start_frame + processed,
            width,
            height,
            fps,
            confirmed_track_ids,
            args.confirmed_only,
            args.display_conf,
        )

        if detections_path is None:
            low_confidence_association_available = (
                detection_confidence <= tracker_config.track_low_thresh
            )
        elif detection_source_confidence is None:
            low_confidence_association_available = None
        else:
            low_confidence_association_available = (
                detection_confidence <= tracker_config.track_low_thresh
                and detection_source_confidence
                <= tracker_config.track_low_thresh
            )
        summary = {
            "tracking_run_id": run_id,
            "input_video": str(input_path),
            "detection_source": (
                str(detections_path)
                if detections_path is not None
                else "fresh Ultralytics YOLO inference"
            ),
            "model": str(model_path) if detections_path is None else detection_metadata.get("model"),
            "tracker": "Ultralytics BYTETracker (ByteTrack)",
            "tracker_config": (
                "Ultralytics cfg/trackers/bytetrack.yaml defaults with CLI overrides"
            ),
            "tracker_thresholds": {
                "track_low_thresh": tracker_config.track_low_thresh,
                "track_high_thresh": tracker_config.track_high_thresh,
                "new_track_thresh": tracker_config.new_track_thresh,
            },
            "tracker_parameters": {
                "detection_input_confidence": detection_confidence,
                "precomputed_source_confidence": detection_source_confidence,
                "low_confidence_association_available": (
                    low_confidence_association_available
                ),
                "device": device,
                "random_seed": RANDOM_SEED,
            },
            "confirmed_candidate_policy": {
                "minimum_distinct_frames": args.min_confirmed_frames,
                "confirmed_track_ids": sorted(confirmed_track_ids),
                "confirmed_track_count": len(confirmed_track_ids),
                "confirmed_person_detections": confirmed_detection_count,
                "visualization_confirmed_only": args.confirmed_only,
                "visualization_confidence": args.display_conf,
                "box_size_filter_applied": False,
            },
            "video_fps": fps,
            "frame_width": width,
            "frame_height": height,
            "source_frame_count": frame_count,
            "start_frame": start_frame,
            "frames_processed": processed,
            "end_frame_exclusive": start_frame + processed,
            "source_detections_after_confidence_filter": total_source_detections,
            "raw_tracks_csv": "tracks.csv",
            "confirmed_candidates_csv": "tracks_confirmed.csv",
            "annotated_video_mode": (
                "confirmed candidates only" if args.confirmed_only
                else "all raw detections above --display-conf"
            ),
            **statistics.summary(),
            "track_identity_limitation": TRACKING_CAVEAT,
            "sports_ball_tracking": (
                "Sports balls are written as class 32 with blank track_id; "
                "ByteTrack is applied only to person detections."
            ),
            "camera_motion_compatibility": {
                "camera_compensation_applied": False,
                "failed_transitions_in_current_quality_file": count_failed_camera_transitions(
                    ROOT / "data" / "output" / "camera_quality_hybrid.csv"
                ),
                "note": CAMERA_COMPENSATION_NOTE,
            },
            "csv_columns": TRACK_CSV_FIELDS,
        }
        with temporary["summary"].open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")

        published = []
        try:
            for key, filename in (
                ("video", "annotated_tracking.mp4"),
                ("csv", "tracks.csv"),
                ("confirmed_csv", "tracks_confirmed.csv"),
                ("summary", "tracking_summary.json"),
            ):
                final_path = paths[filename]
                if final_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite tracking output: {final_path}"
                    )
                temporary[key].rename(final_path)
                published.append(final_path)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise
        return summary
    except Exception:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args, parser)
        summary = run_tracking(args)
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        csv.Error,
    ) as exc:
        parser.error(str(exc))
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Tracked detections: {summary['total_tracked_detections']}")
    print(f"Unique track IDs: {summary['unique_track_ids']}")
    print(f"Saved video: {args.output_dir / 'annotated_tracking.mp4'}")
    print(f"Saved CSV: {args.output_dir / 'tracks.csv'}")
    print(f"Saved confirmed candidates: {args.output_dir / 'tracks_confirmed.csv'}")
    print(f"Saved summary: {args.output_dir / 'tracking_summary.json'}")


if __name__ == "__main__":
    main()
