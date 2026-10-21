import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "output"

TRACK_COLUMNS = {
    "run_id",
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "confidence",
    "foot_x",
    "foot_y",
}
TEAM_COLUMNS = {
    "run_id",
    "track_id",
    "predicted_team",
    "predicted_role",
    "classification_confidence",
    "number_of_samples",
}
COMPENSATED_COLUMNS = {
    "run_id",
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "camera_segment_id",
    "reference_frame",
    "compensation_valid",
    "compensated_x",
    "compensated_y",
}
MAPPING_COLUMNS = {
    "run_id",
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "camera_segment_id",
    "pitch_x_m",
    "pitch_y_m",
    "pitch_mapping_valid",
    "pitch_mapping_invalid_reason",
}
NUMERIC_COLUMNS = (
    "frame",
    "timestamp",
    "class_id",
    "confidence",
    "foot_x",
    "foot_y",
    "compensated_x",
    "compensated_y",
    "reference_frame",
    "pitch_x_m",
    "pitch_y_m",
    "classification_confidence",
    "number_of_samples",
    "classification_confidence",
    "number_of_samples",
)


class DataValidationError(ValueError):
    """Raised when source files are present but cannot be interpreted safely."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    input_dir: Path
    output_dir: Path

    @classmethod
    def from_environment(cls, root=ROOT):
        root = Path(root).expanduser().resolve()
        data_dir = Path(os.environ.get("FMI_DATA_DIR", root / "data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        data_dir = data_dir.resolve()
        input_dir = Path(
            os.environ.get("FMI_INPUT_DIR", data_dir / "input")
        ).expanduser()
        output_dir = Path(
            os.environ.get("FMI_OUTPUT_DIR", data_dir / "output")
        ).expanduser()
        if not input_dir.is_absolute():
            input_dir = root / input_dir
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        return cls(root, data_dir, input_dir.resolve(), output_dir.resolve())


def read_json(path, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required JSON file is missing: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"Expected a JSON object in {path}.")
    return value


def read_csv(path, required_columns=(), required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required CSV file is missing: {path}")
        return None
    try:
        frame = pd.read_csv(path, dtype={"run_id": "string", "track_id": "string"})
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataValidationError(f"Could not read CSV {path}: {exc}") from exc
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"{path} is missing required column(s): {', '.join(sorted(missing))}."
        )
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("run_id", "track_id"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def discover_tracking_runs(output_dir):
    output_dir = Path(output_dir)
    runs = []
    if not output_dir.is_dir():
        return runs
    for summary_path in output_dir.glob("*/tracking_summary.json"):
        try:
            summary = read_json(summary_path, required=True)
        except DataValidationError:
            continue
        tracks_path = summary_path.parent / "tracks_confirmed.csv"
        if tracks_path.is_file():
            runs.append({
                "label": (
                    f"{summary_path.parent.name} — "
                    f"{summary.get('tracking_run_id', 'unknown run')}"
                ),
                "directory": summary_path.parent,
                "summary": summary,
                "tracks_path": tracks_path,
            })
    runs.sort(
        key=lambda item: (
            item["directory"].name != "tracking_confirmed_test",
            item["directory"].name.lower(),
        )
    )
    return runs


def _compatible_summary_paths(output_dir, pattern, run_id, video_path):
    matches = []
    for summary_path in Path(output_dir).glob(pattern):
        try:
            summary = read_json(summary_path, required=True)
        except DataValidationError:
            continue
        if (
            summary.get("tracking_run_id") == run_id
            and _same_video(summary.get("input_video"), video_path)
        ):
            matches.append((summary_path, summary))
    return sorted(matches, key=lambda item: str(item[0]).lower())


def _same_video(recorded, selected):
    if not recorded or not selected:
        return False
    try:
        return Path(recorded).expanduser().resolve() == Path(selected).expanduser().resolve()
    except (OSError, RuntimeError, TypeError):
        return False


def _frame_span(summary):
    recorded_range = summary.get("tracking_frame_range")
    if isinstance(recorded_range, (list, tuple)) and len(recorded_range) == 2:
        try:
            start, end = int(recorded_range[0]), int(recorded_range[1])
            return (start, end) if end >= start else None
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        start = int(summary.get("start_frame", 0))
        end = summary.get("end_frame_exclusive")
        if end is None and summary.get("frames_processed") is not None:
            end = start + int(summary["frames_processed"])
        return (start, int(end)) if end is not None and int(end) >= start else None
    except (TypeError, ValueError, OverflowError):
        return None


def _unique_run_ids(frame, label):
    if frame is None or frame.empty:
        return set()
    ids = set(frame["run_id"].dropna().astype(str)) - {""}
    if len(ids) > 1:
        raise DataValidationError(f"{label} contains multiple tracking run UUIDs.")
    return ids


def _normalize_boolean(series):
    values = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    return values.map(mapping).astype("boolean")


def _with_track_assignments(tracks, assignments, warnings):
    if tracks is None:
        return None
    tracks = tracks.copy()
    tracks["track_id"] = tracks["track_id"].astype(str)
    tracks["run_id"] = tracks["run_id"].astype(str)
    if assignments is None:
        tracks["predicted_team"] = "unknown"
        tracks["predicted_role"] = "unknown"
        tracks["classification_confidence"] = np.nan
        tracks["number_of_samples"] = np.nan
        return tracks
    duplicates = assignments.duplicated(["run_id", "track_id"], keep=False)
    if duplicates.any():
        raise DataValidationError("Team assignment file has duplicate run/track keys.")
    merged = tracks.merge(
        assignments,
        on=["run_id", "track_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_assignment"),
        indicator=True,
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        warnings.append(
            f"{missing} tracking observation(s) have no run-scoped team assignment."
        )
    merged["predicted_team"] = merged["predicted_team"].fillna("unknown")
    merged["predicted_role"] = merged["predicted_role"].fillna("unknown")
    return merged.drop(columns="_merge")


def load_dashboard_dataset(output_dir, tracking_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    tracking_dir = Path(tracking_dir).expanduser().resolve()
    warnings = []
    errors = []
    tracking_summary_path = tracking_dir / "tracking_summary.json"
    tracks_path = tracking_dir / "tracks_confirmed.csv"
    raw_tracks_path = tracking_dir / "tracks.csv"
    tracking_summary = read_json(tracking_summary_path)
    if tracking_summary is None:
        raise FileNotFoundError(f"Tracking summary is missing: {tracking_summary_path}")
    run_id = str(tracking_summary.get("tracking_run_id", "")).strip()
    video_path = tracking_summary.get("input_video")
    if not run_id or not video_path:
        raise DataValidationError(
            "Tracking summary must provide tracking_run_id and input_video."
        )
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Confirmed tracking CSV is missing: {tracks_path}")

    confirmed = read_csv(tracks_path, TRACK_COLUMNS, required=True)
    raw = read_csv(raw_tracks_path, TRACK_COLUMNS) if raw_tracks_path.is_file() else None
    if raw is None:
        warnings.append("Raw tracking CSV is unavailable; provisional-track counts use confirmed data.")
    for label, frame in (("Confirmed tracks", confirmed), ("Raw tracks", raw)):
        if frame is not None and _unique_run_ids(frame, label) - {run_id}:
            raise DataValidationError(f"{label} does not match the selected tracking run UUID.")
        if frame is not None:
            observed_runs = _unique_run_ids(frame, label)
            if observed_runs and observed_runs != {run_id}:
                raise DataValidationError(
                    f"{label} contains an inconsistent tracking run UUID."
                )

    team_matches = _compatible_summary_paths(
        output_dir, "*/team_classification_summary.json", run_id, video_path
    )
    team_matches = [
        item for item in team_matches
        if _same_video(
            item[1].get("tracking_csv"),
            (tracking_dir / "tracks_confirmed.csv"),
        )
    ]
    team_summary_path, team_summary = team_matches[-1] if team_matches else (None, None)
    assignments = None
    if team_summary_path:
        team_csv_path = team_summary_path.parent / "team_assignments.csv"
        assignments = read_csv(team_csv_path, TEAM_COLUMNS, required=True)
        _unique_run_ids(assignments, "Team assignments")
        if _unique_run_ids(assignments, "Team assignments") - {run_id}:
            raise DataValidationError("Team assignments do not match selected tracking UUID.")
        if assignments.duplicated(["run_id", "track_id"]).any():
            raise DataValidationError("Team assignment run/track keys are not unique.")
    else:
        warnings.append("No team-classification output matches this tracking run and video.")

    compensated_matches = _compatible_summary_paths(
        output_dir, "**/compensation_summary.json", run_id, video_path
    )
    compensated_matches = [
        item for item in compensated_matches
        if _same_video(
            item[1].get("tracking_csv"),
            (tracking_dir / "tracks_confirmed.csv"),
        )
    ]
    compensation_summary_path, compensation_summary = (
        compensated_matches[-1] if compensated_matches else (None, None)
    )
    compensated = None
    if compensation_summary_path:
        compensated_path = compensation_summary_path.parent / "compensated_tracks.csv"
        compensated = read_csv(compensated_path, COMPENSATED_COLUMNS, required=True)
        if _unique_run_ids(compensated, "Compensated tracks") - {run_id}:
            raise DataValidationError("Compensated tracks do not match selected run UUID.")
    else:
        warnings.append("No camera-compensation output matches this tracking run and video.")

    all_mapping_matches = _compatible_summary_paths(
        output_dir, "**/mapping_summary.json", run_id, video_path
    )
    mapping_matches = list(all_mapping_matches)
    if compensation_summary:
        mapping_matches = [
            item for item in mapping_matches
            if item[1].get("compensation_run_id")
            == compensation_summary.get("compensation_run_id")
            and _same_video(
                item[1].get("compensated_tracking_csv"),
                compensation_summary_path.parent / "compensated_tracks.csv",
            )
        ]
    mapping_summary_path, mapping_summary = mapping_matches[-1] if mapping_matches else (None, None)
    mapping = None
    if mapping_summary_path:
        mapping_path = mapping_summary_path.parent / "mapped_player_positions.csv"
        mapping = read_csv(mapping_path, MAPPING_COLUMNS, required=True)
        if _unique_run_ids(mapping, "Pitch mapping") - {run_id}:
            raise DataValidationError("Pitch mapping rows do not match selected run UUID.")
        if compensation_summary and mapping_summary.get("compensation_run_id") != compensation_summary.get("compensation_run_id"):
            errors.append("Mapping and selected compensation summaries have different run IDs.")
            mapping = None
    else:
        if compensation_summary and all_mapping_matches:
            errors.append(
                "Mapping and selected compensation summaries have different run IDs or input files."
            )
        warnings.append("No pitch-mapping output matches this tracking run and video.")

    detection_matches = []
    same_video_detections = []
    tracking_span = _frame_span(tracking_summary)
    for path in output_dir.glob("*/detection_summary.json"):
        summary = read_json(path)
        if summary and _same_video(summary.get("input_video"), video_path):
            same_video_detections.append((path, summary))
            if tracking_span is not None and _frame_span(summary) == tracking_span:
                detection_matches.append((path, summary))
    detection_summary_path, detection_summary = (
        sorted(detection_matches, key=lambda item: str(item[0]).lower())[-1]
        if detection_matches else (None, None)
    )
    detections = None
    if detection_summary_path:
        detections = read_csv(
            detection_summary_path.parent / "detections.csv",
            {"frame_number", "object_class", "confidence", "x1", "y1", "x2", "y2"},
        )
    else:
        if same_video_detections:
            warnings.append(
                "Detection outputs exist for this video, but none match the selected tracking frame range."
            )
        else:
            warnings.append("No detection output matches this tracking run and video.")

    camera_quality_path = output_dir / "camera_quality_hybrid.csv"
    camera_quality = read_csv(camera_quality_path, {"frame", "method", "valid"})
    if camera_quality is None:
        warnings.append("Hybrid camera-quality CSV is unavailable.")
    landmark_validation = read_csv(
        output_dir / "landmark_validation" / "validation_summary.csv",
        {"test_frame", "landmark_count", "median_error_px", "max_error_px", "p90_error_px"},
    )
    calibration = None
    calibration_paths = sorted(output_dir.glob("pitch_mapping/**/pitch_calibration.json"))
    for calibration_path in calibration_paths:
        candidate = read_json(calibration_path)
        if (
            candidate
            and candidate.get("tracking_run_id") == run_id
            and candidate.get("camera_segment_id") in (
                mapping_summary.get("calibrated_camera_segments", [])
                if mapping_summary else []
            )
            and candidate.get("valid_for_mapping")
        ):
            calibration = candidate
            break

    confirmed = _with_track_assignments(confirmed, assignments, warnings)
    if raw is not None:
        raw = _with_track_assignments(raw, assignments, [])
    if compensated is not None:
        compensated = _with_track_assignments(compensated, assignments, warnings)
    if mapping is not None and assignments is not None:
        mapping = mapping.merge(
            assignments,
            on=["run_id", "track_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_assignment"),
        )
        mapping["predicted_team"] = mapping["predicted_team"].fillna("unknown")
        mapping["predicted_role"] = mapping["predicted_role"].fillna("unknown")

    return {
        "run_id": run_id,
        "video_path": Path(video_path),
        "tracking_dir": tracking_dir,
        "tracking_summary": tracking_summary,
        "confirmed_tracks": confirmed,
        "raw_tracks": raw,
        "assignments": assignments,
        "team_summary": team_summary,
        "compensated_tracks": compensated,
        "compensation_summary": compensation_summary,
        "mapping": mapping,
        "mapping_summary": mapping_summary,
        "detections": detections,
        "detection_summary": detection_summary,
        "camera_quality": camera_quality,
        "landmark_validation": landmark_validation,
        "calibration": calibration,
        "camera_quality_path": camera_quality_path,
        "warnings": warnings,
        "errors": errors,
        "source_summaries": {
            "tracking": str(tracking_summary_path),
            "teams": str(team_summary_path) if team_summary_path else None,
            "compensation": str(compensation_summary_path) if compensation_summary_path else None,
            "mapping": str(mapping_summary_path) if mapping_summary_path else None,
            "detection": str(detection_summary_path) if detection_summary_path else None,
        },
    }


def duration_seconds(summary):
    fps = _finite_positive(summary.get("video_fps", summary.get("source_video_fps")))
    if fps is None:
        return None
    start = summary.get("start_frame", 0)
    end = summary.get("end_frame_exclusive")
    if end is None:
        processed = summary.get("frames_processed")
        end = start + processed if processed is not None else None
    try:
        frames = int(end) - int(start)
    except (TypeError, ValueError):
        return None
    if frames < 0:
        return None
    return frames / fps


def person_detection_count(detections):
    if detections is None or "object_class" not in detections.columns:
        return None
    classes = detections["object_class"].astype("string").str.strip().str.lower()
    return int(classes.eq("person").sum())


def _finite_positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def track_run_summary(tracks):
    if tracks is None or tracks.empty:
        return {
            "observation_count": 0,
            "unique_track_count": 0,
            "frame_count": 0,
            "first_frame": None,
            "last_frame": None,
        }
    people = tracks[tracks["class_id"] == 0].copy()
    frames = people["frame"].dropna()
    return {
        "observation_count": int(len(people)),
        "unique_track_count": int(people["track_id"].nunique()),
        "frame_count": int(frames.nunique()),
        "first_frame": int(frames.min()) if not frames.empty else None,
        "last_frame": int(frames.max()) if not frames.empty else None,
    }


def player_track_analytics(tracks, track_id):
    if tracks is None or tracks.empty:
        return {"observations": pd.DataFrame(), "summary": None}
    selected = tracks[
        (tracks["class_id"] == 0)
        & (tracks["track_id"].astype(str) == str(track_id))
    ].copy()
    if selected.empty:
        return {"observations": selected, "summary": None}
    runs = selected["run_id"].dropna().astype(str).unique()
    if len(runs) != 1:
        raise DataValidationError("A selected tracking ID spans multiple tracking runs.")
    selected = selected.sort_values(["frame", "timestamp"]).drop_duplicates("frame")
    frames = selected["frame"].dropna().astype(int).to_numpy()
    gaps = np.maximum(np.diff(frames) - 1, 0)
    confidence = pd.to_numeric(selected["confidence"], errors="coerce").dropna()
    summary = {
        "run_id": runs[0],
        "track_id": str(track_id),
        "observations": int(len(selected)),
        "first_frame": int(frames.min()),
        "last_frame": int(frames.max()),
        "missing_frame_count": int(gaps.sum()),
        "gap_count": int((gaps > 0).sum()),
        "largest_gap_frames": int(gaps.max()) if gaps.size else 0,
        "mean_confidence": float(confidence.mean()) if not confidence.empty else None,
        "median_confidence": float(confidence.median()) if not confidence.empty else None,
        "minimum_confidence": float(confidence.min()) if not confidence.empty else None,
        "maximum_confidence": float(confidence.max()) if not confidence.empty else None,
        "fragmentation_note": (
            "Gaps indicate missing observations, not proof of an identity switch."
        ),
    }
    return {"observations": selected, "summary": summary}


def image_heatmap(frame, x_column, y_column, bins=(64, 36), bounds=None):
    if frame is None or frame.empty:
        return np.zeros((bins[1], bins[0]), dtype=np.float64), bounds or (0, 1, 0, 1), 0
    if x_column not in frame.columns or y_column not in frame.columns:
        raise DataValidationError(f"Heatmap requires {x_column} and {y_column}.")
    points = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if bounds is None:
        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max()) if len(points) else 1
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max()) if len(points) else 1
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        bounds = (x_min, x_max, y_min, y_max)
    if len(points) == 0:
        return np.zeros((bins[1], bins[0]), dtype=np.float64), bounds, 0
    heat, _, _ = np.histogram2d(
        points[:, 1], points[:, 0],
        bins=(bins[1], bins[0]),
        range=((bounds[2], bounds[3]), (bounds[0], bounds[1])),
    )
    return heat, bounds, int(len(points))


def valid_pitch_rows(mapping):
    if mapping is None or mapping.empty:
        return mapping.iloc[0:0].copy() if mapping is not None else pd.DataFrame()
    valid = _normalize_boolean(mapping["pitch_mapping_valid"]).fillna(False)
    x = pd.to_numeric(mapping["pitch_x_m"], errors="coerce")
    y = pd.to_numeric(mapping["pitch_y_m"], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y)
    return mapping[valid & finite].copy()


def pitch_mapping_reason_counts(mapping):
    if mapping is None or mapping.empty:
        return pd.DataFrame(columns=["Reason", "Observations"])
    valid = _normalize_boolean(mapping["pitch_mapping_valid"]).fillna(False)
    invalid = mapping.loc[~valid].copy()
    if "class_id" in invalid.columns:
        invalid = invalid[pd.to_numeric(invalid["class_id"], errors="coerce") == 0]
    if invalid.empty:
        return pd.DataFrame(columns=["Reason", "Observations"])
    reasons = invalid["pitch_mapping_invalid_reason"].fillna("").astype(str).str.strip()
    reasons = reasons.mask(reasons.eq(""), "Not specified")
    return (
        reasons.value_counts()
        .rename_axis("Reason")
        .reset_index(name="Observations")
    )


def landmark_validation_views(validation):
    columns = ["test_frame", "median_error_px", "max_error_px", "run_id", "review_state"]
    if validation is None or validation.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), False

    history = validation.copy()
    if "run_id" not in history.columns:
        history["run_id"] = "run-id-unavailable"
    history["test_frame"] = pd.to_numeric(history["test_frame"], errors="coerce")
    for column in ("median_error_px", "max_error_px"):
        history[column] = pd.to_numeric(history[column], errors="coerce")

    review_column = next(
        (
            name for name in (
                "review_status", "is_reviewed", "reviewed", "reviewed_at"
            ) if name in history.columns
        ),
        None,
    )
    review_metadata_available = review_column is not None
    if review_column is None:
        reviewed = pd.Series(False, index=history.index)
    else:
        if review_column == "reviewed_at":
            reviewed = pd.to_datetime(
                history[review_column], errors="coerce", utc=True
            ).notna()
        else:
            reviewed = (
                history[review_column]
                .astype("string")
                .str.strip()
                .str.casefold()
                .isin({"true", "1", "yes", "reviewed", "approved", "accepted"})
            )
    history["_reviewed"] = reviewed.fillna(False)

    if "created_at" in history.columns:
        history["_run_order"] = pd.to_datetime(
            history["created_at"], errors="coerce", utc=True
        )
    else:
        history["_run_order"] = pd.NaT
    history["_row_order"] = np.arange(len(history))
    history = history.sort_values(
        ["_run_order", "_row_order"], na_position="first", kind="stable"
    )

    latest_indices = []
    for _, group in history.dropna(subset=["test_frame"]).groupby("test_frame", sort=True):
        reviewed_group = group[group["_reviewed"]]
        latest_indices.append(
            (reviewed_group if not reviewed_group.empty else group).index[-1]
        )
    history["review_state"] = (
        "Unreviewed; excluded from reviewed trend"
        if review_metadata_available else "Unreviewed; superseded"
    )
    history.loc[history["_reviewed"], "review_state"] = "Reviewed; superseded"
    for index in latest_indices:
        history.loc[index, "review_state"] = (
            "Latest reviewed" if bool(history.loc[index, "_reviewed"])
            else "Latest recorded; unreviewed"
        )
    latest = history.loc[latest_indices].copy() if latest_indices else history.iloc[0:0].copy()
    visible_columns = [name for name in history.columns if not name.startswith("_")]
    return (
        latest[visible_columns].sort_values("test_frame").reset_index(drop=True),
        history.sort_values(
            ["test_frame", "_run_order", "_row_order"], na_position="last", kind="stable"
        )[visible_columns].reset_index(drop=True),
        review_metadata_available,
    )


def camera_segment_rows(compensated, segment_id):
    if compensated is None or compensated.empty:
        return compensated.iloc[0:0].copy() if compensated is not None else pd.DataFrame()
    return compensated[
        (compensated["camera_segment_id"].astype(str) == str(segment_id))
        & _normalize_boolean(compensated["compensation_valid"]).fillna(False)
    ].copy()


def available_videos(output_dir):
    root = Path(output_dir)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.mp4"), key=lambda path: str(path).lower())


def compatible_video_options(output_dir, dataset):
    """Return selected-run video variants, preferring H.264 and retaining originals."""
    output_dir = Path(output_dir)
    source_video = Path(dataset["video_path"]).expanduser()
    tracking_summary = dataset.get("tracking_summary") or {}
    run_id = str(dataset.get("run_id", ""))
    tracking_span = _frame_span(tracking_summary)
    tracking_csv = Path(dataset.get("tracking_dir", "")) / "tracks_confirmed.csv"
    summaries = dataset.get("source_summaries") or {}
    browser_dir = output_dir / "browser_videos"

    def summary_path(key):
        value = summaries.get(key)
        return Path(value) if value else None

    def basic_compatibility(summary, *, require_run=True, frame_range=None):
        return bool(
            summary
            and _same_video(summary.get("input_video"), source_video)
            and (not require_run or summary.get("tracking_run_id") == run_id)
            and (frame_range is None or _frame_span(summary) == frame_range)
        )

    specifications = []

    detection_path = summary_path("detection")
    detection = dataset.get("detection_summary")
    if (
        detection_path
        and basic_compatibility(detection, require_run=False, frame_range=tracking_span)
    ):
        specifications.append((
            "YOLO detection",
            detection_path.parent / "annotated_detection.mp4",
            "annotated_detection_h264.mp4",
        ))

    tracking_path = summary_path("tracking")
    if (
        tracking_path
        and basic_compatibility(tracking_summary, frame_range=tracking_span)
        and _same_video(tracking_summary.get("input_video"), source_video)
    ):
        specifications.append((
            "ByteTrack tracking",
            tracking_path.parent / "annotated_tracking.mp4",
            "annotated_tracking_h264.mp4",
        ))

    team_path = summary_path("teams")
    team_summary = dataset.get("team_summary")
    if (
        team_path
        and basic_compatibility(team_summary, frame_range=tracking_span)
        and _same_video(team_summary.get("tracking_csv"), tracking_csv)
    ):
        specifications.append((
            "Team classification",
            team_path.parent / "annotated_teams.mp4",
            "annotated_teams_h264.mp4",
        ))

    compensation_path = summary_path("compensation")
    compensation_summary = dataset.get("compensation_summary")
    if (
        compensation_path
        and basic_compatibility(compensation_summary, frame_range=tracking_span)
        and _same_video(compensation_summary.get("tracking_csv"), tracking_csv)
    ):
        specifications.append((
            "Camera-compensation comparison",
            compensation_path.parent / "trajectory_comparison.mp4",
            "trajectory_comparison_h264.mp4",
        ))

    mapping_path = summary_path("mapping")
    mapping_summary = dataset.get("mapping_summary")
    compensation_csv = (
        compensation_path.parent / "compensated_tracks.csv"
        if compensation_path else None
    )
    if (
        mapping_path
        and basic_compatibility(mapping_summary)
        and compensation_summary
        and mapping_summary.get("compensation_run_id")
        == compensation_summary.get("compensation_run_id")
        and compensation_csv is not None
        and _same_video(mapping_summary.get("compensated_tracking_csv"), compensation_csv)
        and _frame_span(compensation_summary) == tracking_span
    ):
        specifications.append((
            "Pitch mapping",
            mapping_path.parent / "annotated_pitch_mapping.mp4",
            "annotated_pitch_mapping_h264.mp4",
        ))

    options = []
    for label, original, h264_name in specifications:
        h264 = browser_dir / h264_name
        if h264.is_file():
            options.append({"path": h264.resolve(), "label": f"{label} · H.264"})
    for label, original, _ in specifications:
        if original.is_file():
            options.append({"path": original.resolve(), "label": f"{label} · original"})
    if source_video.is_file():
        options.append({
            "path": source_video.resolve(),
            "label": f"Source video · {source_video.name}",
        })
    return options


def order_video_options(videos, preferred=(), original=None):
    available = []
    seen = set()
    for path in [*preferred, *videos, original]:
        if path is None:
            continue
        candidate = Path(path).expanduser()
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.is_file() and resolved not in seen:
            available.append(resolved)
            seen.add(resolved)
    preferred_paths = []
    for path in preferred:
        if path is None:
            continue
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen and resolved not in preferred_paths:
            preferred_paths.append(resolved)
    original_path = None
    if original is not None:
        try:
            candidate = Path(original).expanduser().resolve()
            if candidate in seen:
                original_path = candidate
        except (OSError, RuntimeError):
            pass
    return (
        preferred_paths
        + [path for path in available if path not in preferred_paths and path != original_path]
        + ([original_path] if original_path is not None else [])
    )


def display_path(path, root):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except (OSError, RuntimeError, ValueError):
        return str(path)


def diagnostic_metrics(dataset):
    summary = dataset.get("tracking_summary", {})
    confirmed = dataset.get("confirmed_tracks")
    raw = dataset.get("raw_tracks")
    compensation_summary = dataset.get("compensation_summary") or {}
    mapping_summary = dataset.get("mapping_summary") or {}
    assignments = dataset.get("assignments")
    roles = (
        assignments["predicted_role"].fillna("unknown").astype(str).value_counts().to_dict()
        if assignments is not None and "predicted_role" in assignments
        else {}
    )
    return {
        "tracking_run_id": dataset.get("run_id"),
        "frames_analyzed": summary.get("frames_processed"),
        "video_duration_seconds": duration_seconds(summary),
        "raw_tracking": track_run_summary(raw),
        "confirmed_tracking": track_run_summary(confirmed),
        "team_assignment_count": int(len(assignments)) if assignments is not None else None,
        "role_counts": roles,
        "camera_segments": compensation_summary.get("camera_segments", []),
        "failed_camera_transitions": compensation_summary.get("failed_transitions"),
        "mapping": mapping_summary,
        "data_warnings": list(dataset.get("warnings", [])),
        "data_errors": list(dataset.get("errors", [])),
    }
