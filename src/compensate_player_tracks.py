import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKS = ROOT / "data" / "output" / "tracking_confirmed_test" / "tracks_confirmed.csv"
DEFAULT_TRACK_SET = "confirmed"
DEFAULT_TRANSFORMS = ROOT / "data" / "output" / "camera_transforms_hybrid.npy"
DEFAULT_QUALITY = ROOT / "data" / "output" / "camera_quality_hybrid.csv"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "compensated_tracking"
DEFAULT_CONDITION_LIMIT = 1e12
DENOMINATOR_EPSILON = 1e-12
PANEL_WIDTH = 960
PANEL_HEIGHT = 540
OUTPUT_NAMES = (
    "compensated_tracks.csv",
    "compensation_summary.json",
    "trajectory_comparison.mp4",
    "trajectory_preview.png",
)
COMPENSATION_FIELDS = [
    "compensated_x",
    "compensated_y",
    "camera_segment_id",
    "reference_frame",
    "compensation_valid",
    "invalid_reason",
    "segment_start_reason",
]


@dataclass(frozen=True)
class FrameTransform:
    segment_id: str
    reference_frame: int
    matrix: np.ndarray | None
    valid: bool
    invalid_reason: str
    segment_start_reason: str


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Map tracked person foot points into piecewise camera-reference "
            "coordinates. Distinct camera segments are not directly comparable."
        )
    )
    parser.add_argument(
        "--input-tracks",
        type=Path,
        required=True,
        help="One run's tracks.csv or tracks_confirmed.csv from track_objects.py.",
    )
    parser.add_argument(
        "--track-set",
        choices=("raw", "confirmed"),
        required=True,
        help="Explicitly identify whether the selected input is raw or confirmed.",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition-limit",
        type=float,
        default=DEFAULT_CONDITION_LIMIT,
        help=f"Maximum matrix condition number (default: {DEFAULT_CONDITION_LIMIT:g}).",
    )
    return parser


def validate_args(args, parser=None):
    if not math.isfinite(args.condition_limit) or args.condition_limit <= 1:
        message = "--condition-limit must be finite and greater than 1."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not args.input_tracks:
        message = "--input-tracks is required."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not args.track_set:
        message = "--track-set must explicitly be raw or confirmed."
        if parser:
            parser.error(message)
        raise ValueError(message)


def _validity_value(value, row_number):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"Invalid camera quality valid flag at row {row_number}: {value!r}."
    )


def load_camera_data(transforms_path, quality_path):
    transforms = np.load(transforms_path, allow_pickle=False)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError(
            f"Transform array must have shape (frames, 3, 3); got {transforms.shape}."
        )
    with quality_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"frame", "method", "valid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{quality_path} must contain frame, method, and valid columns."
            )
        quality = []
        for row_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame number in camera quality row {row_number}."
                ) from exc
            quality.append({
                "frame": frame,
                "method": row["method"].strip().lower(),
                "valid": _validity_value(row["valid"], row_number),
            })
    if len(quality) != len(transforms):
        raise ValueError(
            f"Transform array has {len(transforms)} frames but quality CSV has "
            f"{len(quality)} rows."
        )
    expected_frames = list(range(len(transforms)))
    actual_frames = [item["frame"] for item in quality]
    if actual_frames != expected_frames:
        raise ValueError(
            "Camera quality rows must be ordered and cover each transform frame "
            "exactly once starting at frame 0."
        )
    return transforms.astype(np.float64, copy=False), quality


def _matrix_error(matrix, condition_limit):
    if matrix.shape != (3, 3):
        return "invalid_transform_shape"
    if not np.isfinite(matrix).all():
        return "nonfinite_transform"
    try:
        determinant = float(np.linalg.det(matrix))
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return "singular_transform"
    if not math.isfinite(determinant) or abs(determinant) <= 1e-15:
        return "singular_transform"
    if not math.isfinite(condition) or condition > condition_limit:
        return "ill_conditioned_transform"
    return ""


def build_frame_transforms(transforms, quality, condition_limit=DEFAULT_CONDITION_LIMIT):
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("Transform array must have shape (frames, 3, 3).")
    if len(quality) != len(transforms):
        raise ValueError("Transform and quality frame counts differ.")
    if len(transforms) == 0:
        return []

    identity = np.eye(3, dtype=np.float64)
    frames = []
    segment_number = -1
    reference_frame = 0
    cumulative = identity.copy()
    reset_after_invalid = False
    initial_error = _matrix_error(transforms[0], condition_limit)
    if initial_error:
        segment_number = 0
        frames.append(FrameTransform(
            segment_id="segment_000",
            reference_frame=0,
            matrix=None,
            valid=False,
            invalid_reason=initial_error,
            segment_start_reason="invalid_reference_transform",
        ))
        reset_after_invalid = True
    else:
        segment_number = 0
        frames.append(FrameTransform(
            segment_id="segment_000",
            reference_frame=0,
            matrix=identity,
            valid=True,
            invalid_reason="",
            segment_start_reason="initial_reference",
        ))

    for frame_number in range(1, len(transforms)):
        item = quality[frame_number]
        is_failed_transition = not item["valid"] or item["method"] == "failed"
        if is_failed_transition:
            segment_number += 1
            reference_frame = frame_number
            cumulative = identity.copy()
            reset_after_invalid = False
            frames.append(FrameTransform(
                segment_id=f"segment_{segment_number:03d}",
                reference_frame=reference_frame,
                matrix=identity.copy(),
                valid=True,
                invalid_reason="",
                segment_start_reason="failed_transition_new_reference",
            ))
            continue

        if reset_after_invalid:
            segment_number += 1
            reference_frame = frame_number
            cumulative = identity.copy()
            reset_after_invalid = False
            frames.append(FrameTransform(
                segment_id=f"segment_{segment_number:03d}",
                reference_frame=reference_frame,
                matrix=identity.copy(),
                valid=True,
                invalid_reason="",
                segment_start_reason="previous_invalid_transform_new_reference",
            ))
            continue

        step = transforms[frame_number]
        error = _matrix_error(step, condition_limit)
        if error:
            frames.append(FrameTransform(
                segment_id=f"segment_{segment_number:03d}",
                reference_frame=reference_frame,
                matrix=None,
                valid=False,
                invalid_reason=error,
                segment_start_reason="",
            ))
            reset_after_invalid = True
            continue

        candidate = cumulative @ step
        error = _matrix_error(candidate, condition_limit)
        if error:
            frames.append(FrameTransform(
                segment_id=f"segment_{segment_number:03d}",
                reference_frame=reference_frame,
                matrix=None,
                valid=False,
                invalid_reason=f"invalid_transform_chain:{error}",
                segment_start_reason="",
            ))
            reset_after_invalid = True
            continue

        cumulative = candidate
        frames.append(FrameTransform(
            segment_id=f"segment_{segment_number:03d}",
            reference_frame=reference_frame,
            matrix=cumulative.copy(),
            valid=True,
            invalid_reason="",
            segment_start_reason="",
        ))
    return frames


def project_point(matrix, x, y, denominator_epsilon=DENOMINATOR_EPSILON):
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None, "nonfinite_foot_point"
    point = matrix @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if not np.isfinite(point).all():
        return None, "nonfinite_projected_point"
    denominator = float(point[2])
    scale = max(1.0, float(np.linalg.norm(matrix[2, :])))
    if abs(denominator) <= denominator_epsilon * scale:
        return None, "near_zero_homogeneous_denominator"
    projected = point[:2] / denominator
    if not np.isfinite(projected).all():
        return None, "nonfinite_projected_coordinates"
    return (float(projected[0]), float(projected[1])), ""


def _load_tracking_metadata(tracks_path, expected_track_set, video_override=None):
    summary_path = tracks_path.with_name("tracking_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Tracking summary not found: {summary_path}; run provenance is required."
        )
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    expected_file = "tracks.csv" if expected_track_set == "raw" else "tracks_confirmed.csv"
    if tracks_path.name.lower() != expected_file:
        raise ValueError(
            f"--track-set {expected_track_set} requires a file named "
            f"{expected_file}; received {tracks_path.name}."
        )
    if summary.get("raw_tracks_csv") != "tracks.csv":
        raise ValueError("Tracking summary does not identify the canonical raw CSV.")
    if (
        expected_track_set == "confirmed"
        and summary.get("confirmed_candidates_csv") != "tracks_confirmed.csv"
    ):
        raise ValueError(
            "Tracking summary does not identify the canonical confirmed CSV."
        )
    run_id = summary.get("tracking_run_id")
    if not run_id:
        raise ValueError("Tracking summary has no tracking_run_id.")
    video_path = (
        video_override.expanduser().resolve()
        if video_override is not None
        else Path(summary["input_video"]).expanduser().resolve()
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if (
        video_override is not None
        and video_path != Path(summary["input_video"]).expanduser().resolve()
    ):
        raise ValueError("--video does not match the input video recorded in summary.")
    try:
        start_frame = int(summary["start_frame"])
        end_frame = int(summary["end_frame_exclusive"])
        fps = float(summary["video_fps"])
        width = int(summary["frame_width"])
        height = int(summary["frame_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tracking summary is missing valid video/frame metadata.") from exc
    if (
        start_frame < 0
        or end_frame <= start_frame
        or not math.isfinite(fps)
        or fps <= 0
        or width <= 0
        or height <= 0
    ):
        raise ValueError("Tracking summary contains invalid video/frame metadata.")
    return summary, run_id, video_path, start_frame, end_frame, fps, width, height


def read_tracking_rows(
    tracks_path,
    expected_run_id,
    frame_count,
    start_frame=0,
    end_frame=None,
    expected_fps=None,
):
    if end_frame is None:
        end_frame = frame_count
    with tracks_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "run_id", "frame", "timestamp", "track_id", "class_id",
            "x1", "y1", "x2", "y2", "foot_x", "foot_y",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{tracks_path} must contain: " + ", ".join(sorted(required))
            )
        fields = list(reader.fieldnames)
        if any(field in COMPENSATION_FIELDS for field in fields):
            raise ValueError("Input tracking CSV already contains compensation columns.")
        rows = []
        previous_frame = -1
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != expected_run_id:
                raise ValueError(
                    f"Run ID mismatch in tracking CSV row {row_number}; refusing "
                    "to mix tracking runs."
                )
            try:
                frame = int(row["frame"])
                timestamp = float(row["timestamp"])
                class_id = int(row["class_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame, timestamp, or class at CSV row {row_number}."
                ) from exc
            if frame < 0 or frame >= frame_count:
                raise ValueError(
                    f"Tracking CSV frame {frame} at row {row_number} is outside "
                    "the camera transform array."
                )
            if frame < start_frame or frame >= end_frame:
                raise ValueError(
                    f"Tracking CSV frame {frame} at row {row_number} is outside "
                    f"the run range [{start_frame}, {end_frame})."
                )
            if frame < previous_frame:
                raise ValueError("Tracking CSV must be ordered by original frame.")
            if not math.isfinite(timestamp):
                raise ValueError(f"Nonfinite timestamp at tracking row {row_number}.")
            if expected_fps is not None and abs(
                timestamp - frame / expected_fps
            ) > 1e-3:
                raise ValueError(
                    f"Timestamp at tracking row {row_number} does not match "
                    "original frame number / source FPS."
                )
            if row["track_id"] and not row["track_id"].strip():
                raise ValueError(f"Invalid track ID at tracking row {row_number}.")
            rows.append(row)
            previous_frame = frame
    return fields, rows


def compensate_rows(rows, source_fields, frame_transforms):
    output_fields = list(source_fields) + [
        field for field in COMPENSATION_FIELDS if field not in source_fields
    ]
    output = []
    for source in rows:
        row = dict(source)
        frame_number = int(source["frame"])
        frame_transform = frame_transforms[frame_number]
        row["compensated_x"] = ""
        row["compensated_y"] = ""
        row["camera_segment_id"] = frame_transform.segment_id
        row["reference_frame"] = frame_transform.reference_frame
        row["compensation_valid"] = "False"
        row["invalid_reason"] = ""
        row["segment_start_reason"] = frame_transform.segment_start_reason

        if int(source["class_id"]) != 0:
            row["invalid_reason"] = "not_person_detection"
        elif not frame_transform.valid or frame_transform.matrix is None:
            row["invalid_reason"] = frame_transform.invalid_reason
        elif not source["foot_x"] or not source["foot_y"]:
            row["invalid_reason"] = "missing_foot_point"
        else:
            projected, error = project_point(
                frame_transform.matrix, source["foot_x"], source["foot_y"]
            )
            if error:
                row["invalid_reason"] = error
            else:
                row["compensated_x"] = projected[0]
                row["compensated_y"] = projected[1]
                row["compensation_valid"] = "True"
        output.append(row)
    return output_fields, output


def create_output_paths(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite compensation outputs: "
            + ", ".join(existing)
        )
    temporary = {}
    try:
        for name, suffix in (
            ("csv", ".csv"),
            ("summary", ".json"),
            ("video", ".mp4"),
            ("preview", ".png"),
        ):
            descriptor, path = tempfile.mkstemp(
                prefix=f".compensation_{name}_",
                suffix=suffix,
                dir=output_dir,
            )
            os.close(descriptor)
            temporary[name] = Path(path)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return outputs, temporary


def _track_color(track_id):
    value = int(track_id)
    return (
        64 + (value * 73) % 192,
        64 + (value * 131) % 192,
        64 + (value * 197) % 192,
    )


def _raw_panel(frame, rows, raw_trails, segment):
    panel = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT))
    x_scale = PANEL_WIDTH / frame.shape[1]
    y_scale = PANEL_HEIGHT / frame.shape[0]
    for row in rows:
        track_id = row["track_id"]
        class_id = int(row["class_id"])
        if class_id == 32:
            color = (0, 165, 255)
            label = "sports ball (untracked)"
        else:
            color = _track_color(track_id) if track_id else (0, 220, 0)
            label = f"person ID {track_id}" if track_id else "person pending"
            if (
                track_id
                and row["foot_x"] not in (None, "")
                and row["foot_y"] not in (None, "")
            ):
                key = (row["run_id"], track_id)
                foot = (
                    int(round(float(row["foot_x"]) * x_scale)),
                    int(round(float(row["foot_y"]) * y_scale)),
                )
                raw_trails[key].append(foot)
                points = list(raw_trails[key])
                if len(points) > 1:
                    cv2.polylines(
                        panel, [np.asarray(points, dtype=np.int32)], False, color, 2
                    )
        x1 = int(round(float(row["x1"]) * x_scale))
        y1 = int(round(float(row["y1"]) * y_scale))
        x2 = int(round(float(row["x2"]) * x_scale))
        y2 = int(round(float(row["y2"]) * y_scale))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            panel,
            label,
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    segment_id = segment.segment_id if segment else "unavailable"
    reference = segment.reference_frame if segment else "unavailable"
    status = "valid" if segment and segment.valid else "invalid"
    cv2.putText(
        panel,
        f"RAW IMAGE COORDINATES | {segment_id} ref={reference} | {status}",
        (12, PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _compensated_panel(
    rows, frame_number, segment_id, reference_frame, trails, source_width, source_height
):
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, PANEL_HEIGHT - 1), (70, 70, 70), 1)
    cv2.line(panel, (0, PANEL_HEIGHT // 2), (PANEL_WIDTH, PANEL_HEIGHT // 2),
             (40, 40, 40), 1)
    cv2.line(panel, (PANEL_WIDTH // 2, 0), (PANEL_WIDTH // 2, PANEL_HEIGHT),
             (40, 40, 40), 1)
    cv2.putText(
        panel,
        f"COMPENSATED REFERENCE COORDINATES | {segment_id} ref={reference_frame}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    x_scale = PANEL_WIDTH / source_width
    y_scale = PANEL_HEIGHT / source_height
    valid_observations = 0
    for row in rows:
        if (
            int(row["class_id"]) != 0
            or row["compensation_valid"] != "True"
            or row["camera_segment_id"] != segment_id
        ):
            continue
        track_id = row["track_id"]
        if not track_id:
            continue
        key = (row["run_id"], track_id)
        point = (
            int(round(float(row["compensated_x"]) * x_scale)),
            int(round(float(row["compensated_y"]) * y_scale)),
        )
        trails[key].append(point)
        path = list(trails[key])
        color = _track_color(track_id)
        if len(path) > 1:
            cv2.polylines(
                panel, [np.asarray(path, dtype=np.int32)], False, color, 2
            )
        cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            str(track_id),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        valid_observations += 1
    if valid_observations == 0:
        cv2.putText(
            panel,
            "No valid compensated person observations",
            (20, PANEL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "Segment-local pixels; other segments are not comparable",
        (12, PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_comparison_video(
    video_path,
    output_path,
    preview_path,
    start_frame,
    end_frame,
    fps,
    width,
    height,
    observations_by_frame,
    frame_transforms,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    writer = None
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(f"Could not seek to tracking start frame {start_frame}.")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (PANEL_WIDTH * 2, PANEL_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize comparison video: {output_path}")

        raw_trails = defaultdict(lambda: deque(maxlen=30))
        compensated_trails = defaultdict(lambda: deque(maxlen=90))
        previous_segment = None
        preview = None
        progress = tqdm(
            total=end_frame - start_frame,
            desc="Rendering trajectories",
            unit="frame",
        )
        try:
            for frame_number in range(start_frame, end_frame):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read source frame {frame_number} for visualization."
                    )
                transform = frame_transforms[frame_number]
                if previous_segment is not None and transform.segment_id != previous_segment:
                    raw_trails.clear()
                    compensated_trails.clear()
                previous_segment = transform.segment_id
                rows = observations_by_frame.get(frame_number, [])
                left = _raw_panel(frame, rows, raw_trails, transform)
                right = _compensated_panel(
                    rows,
                    frame_number,
                    transform.segment_id,
                    transform.reference_frame,
                    compensated_trails,
                    width,
                    height,
                )
                comparison = cv2.hconcat([left, right])
                writer.write(comparison)
                if frame_number == end_frame - 1:
                    preview = comparison
                progress.update(1)
        finally:
            progress.close()
        if preview is None:
            preview = np.zeros((PANEL_HEIGHT, PANEL_WIDTH * 2, 3), dtype=np.uint8)
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Could not write trajectory preview: {preview_path}")
    finally:
        if writer is not None:
            writer.release()
        capture.release()


def summarize_compensation(rows, frame_transforms, start_frame, end_frame):
    persons = [row for row in rows if int(row["class_id"]) == 0]
    valid = [row for row in persons if row["compensation_valid"] == "True"]
    invalid = [row for row in persons if row["compensation_valid"] != "True"]
    valid_ids = {
        (row["run_id"], row["track_id"])
        for row in valid
        if row["track_id"]
    }
    coordinates = [
        (float(row["compensated_x"]), float(row["compensated_y"]))
        for row in valid
    ]
    maximum_absolute_coordinate = max(
        (max(abs(x), abs(y)) for x, y in coordinates),
        default=0.0,
    )
    invalid_reasons = defaultdict(int)
    for row in invalid:
        invalid_reasons[row["invalid_reason"]] += 1
    encountered = []
    seen = set()
    for frame_number in range(start_frame, end_frame):
        frame_transform = frame_transforms[frame_number]
        if frame_transform.segment_id not in seen:
            seen.add(frame_transform.segment_id)
            encountered.append({
                "camera_segment_id": frame_transform.segment_id,
                "reference_frame": frame_transform.reference_frame,
                "first_frame_in_run": frame_number,
                "start_reason": frame_transform.segment_start_reason,
            })
    boundaries = [
        frame_number
        for frame_number in range(max(1, start_frame), end_frame)
        if frame_transforms[frame_number].segment_start_reason
        == "failed_transition_new_reference"
    ]
    return {
        "person_observations": len(persons),
        "valid_compensated_observations": len(valid),
        "invalid_compensated_observations": len(invalid),
        "compensated_track_ids": len(valid_ids),
        "camera_segments_encountered": len(encountered),
        "camera_segments": encountered,
        "failed_transition_segment_starts_in_run": boundaries,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "maximum_absolute_compensated_coordinate": maximum_absolute_coordinate,
        "extreme_coordinate_observations_over_1e6": sum(
            max(abs(x), abs(y)) > 1e6 for x, y in coordinates
        ),
        "other_segments_coordinate_comparability": (
            "Coordinates in different camera segments use independent reference "
            "frames and are not directly comparable without additional registration."
        ),
    }


def compensate(args):
    validate_args(args)
    tracks_path = args.input_tracks.expanduser().resolve()
    transforms_path = args.transforms.expanduser().resolve()
    quality_path = args.quality.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, label in (
        (tracks_path, "tracking CSV"),
        (transforms_path, "camera transform array"),
        (quality_path, "camera quality CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    tracking_summary, run_id, video_path, start_frame, end_frame, fps, width, height = (
        _load_tracking_metadata(tracks_path, args.track_set, args.video)
    )
    transforms, quality = load_camera_data(transforms_path, quality_path)
    if end_frame > len(transforms):
        raise ValueError(
            f"Tracking summary requests frames through {end_frame - 1}, but "
            f"camera transforms cover only 0 through {len(transforms) - 1}."
        )
    frame_transforms = build_frame_transforms(
        transforms, quality, args.condition_limit
    )
    source_fields, rows = read_tracking_rows(
        tracks_path,
        run_id,
        len(frame_transforms),
        start_frame=start_frame,
        end_frame=end_frame,
        expected_fps=fps,
    )
    csv_fields, compensated_rows = compensate_rows(
        rows, source_fields, frame_transforms
    )
    observations_by_frame = defaultdict(list)
    for row in compensated_rows:
        observations_by_frame[int(row["frame"])].append(row)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if (
        not math.isfinite(actual_fps)
        or abs(actual_fps - fps) > 1e-3
        or actual_width != width
        or actual_height != height
        or end_frame > actual_frames
    ):
        raise ValueError(
            "Input video metadata does not match the tracking summary."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, temporary = create_output_paths(output_dir)
    try:
        with temporary["csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(compensated_rows)

        stats = summarize_compensation(
            compensated_rows, frame_transforms, start_frame, end_frame
        )
        summary = {
            "compensation_run_id": uuid4().hex,
            "tracking_run_id": run_id,
            "track_set": args.track_set,
            "tracking_csv": str(tracks_path),
            "tracking_tracker": tracking_summary.get("tracker"),
            "input_video": str(video_path),
            "camera_transform_file": str(transforms_path),
            "camera_quality_file": str(quality_path),
            "transform_convention": (
                "Hs[f] maps coordinates from frame f to frame f-1. For a "
                "segment reference s, cumulative C[s]=I and C[f]=C[f-1] @ Hs[f]. "
                "Column points [x,y,1] are projected with C[f]."
            ),
            "composition_order": "Cumulative = Hs[s+1] @ ... @ Hs[f]",
            "condition_number_limit": args.condition_limit,
            "homogeneous_denominator_epsilon": DENOMINATOR_EPSILON,
            "source_video_fps": fps,
            "source_frame_width": width,
            "source_frame_height": height,
            "source_frame_count": actual_frames,
            "tracking_frame_range": [start_frame, end_frame],
            "camera_transform_frame_count": len(frame_transforms),
            "failed_camera_transition_count": sum(
                not item["valid"] or item["method"] == "failed"
                for item in quality[1:]
            ),
            "input_tracking_observations": len(rows),
            "non_person_observations_preserved_invalid": sum(
                int(row["class_id"]) != 0 for row in compensated_rows
            ),
            **stats,
            "outputs": {
                "csv": "compensated_tracks.csv",
                "summary": "compensation_summary.json",
                "video": "trajectory_comparison.mp4",
                "preview": "trajectory_preview.png",
            },
        }
        with temporary["summary"].open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")

        render_comparison_video(
            video_path,
            temporary["video"],
            temporary["preview"],
            start_frame,
            end_frame,
            fps,
            width,
            height,
            observations_by_frame,
            frame_transforms,
        )
        for key, filename in (
            ("csv", "compensated_tracks.csv"),
            ("summary", "compensation_summary.json"),
            ("video", "trajectory_comparison.mp4"),
            ("preview", "trajectory_preview.png"),
        ):
            target = outputs[filename]
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite compensation output: {target}"
                )
            temporary[key].replace(target)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = compensate(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        csv.Error,
        np.linalg.LinAlgError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Person observations: "
        f"{summary['person_observations']} "
        f"(valid {summary['valid_compensated_observations']}, "
        f"invalid {summary['invalid_compensated_observations']})"
    )
    print(f"Camera segments: {summary['camera_segments_encountered']}")
    print(f"Compensated track IDs: {summary['compensated_track_ids']}")
    for filename in OUTPUT_NAMES:
        print(f"Saved: {args.output_dir / filename}")


if __name__ == "__main__":
    main()
