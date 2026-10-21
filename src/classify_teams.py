import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKS = (
    ROOT / "data" / "output" / "tracking_confirmed_test" / "tracks_confirmed.csv"
)
DEFAULT_TRACK_SET = "confirmed"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "team_classification"
SAMPLE_EVERY = 3
MIN_SAMPLES = 3
MIN_COLOR_SHARE = 0.24
MIN_COLOR_MARGIN = 0.10
MIN_SPECIAL_ROLE_SAMPLES = 5
HSV_LOW_SATURATION = 45
HSV_LOW_VALUE = 30
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

TEAM_COLORS = {
    "team_a_outfield": (38, 28, 150),
    "team_b_outfield": (220, 90, 30),
    "referee_candidate": (0, 220, 255),
    "goalkeeper_candidate": (40, 210, 40),
    "unknown": (160, 160, 160),
}
DISPLAY_NAMES = {
    "team_a_outfield": "Team A outfield",
    "team_b_outfield": "Team B outfield",
    "referee_candidate": "Referee candidate",
    "goalkeeper_candidate": "Goalkeeper candidate",
    "unknown": "Unknown",
}
OUTPUT_NAMES = (
    "team_assignments.csv",
    "team_classification_summary.json",
    "annotated_teams.mp4",
    "team_classification_preview.png",
)
ASSIGNMENT_FIELDS = [
    "run_id",
    "track_id",
    "predicted_team",
    "predicted_role",
    "classification_confidence",
    "number_of_samples",
    "classification_method",
]
FEATURE_NAMES = (
    "team_a_red_share",
    "team_b_blue_share",
    "referee_yellow_share",
    "goalkeeper_green_share",
)
CLASSIFICATION_METHOD = (
    "median per-track upper-body HSV color shares from sampled detections; "
    "provisional role/color rules with minimum sample, share, and margin checks"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Classify confirmed player tracks from aggregated upper-body jersey "
            "colors. Team and special-role labels are provisional."
        )
    )
    parser.add_argument(
        "--input-tracks",
        type=Path,
        default=DEFAULT_TRACKS,
        help="One run's tracks_confirmed.csv from track_objects.py.",
    )
    parser.add_argument(
        "--track-set",
        choices=("confirmed",),
        default=DEFAULT_TRACK_SET,
        help="This module currently accepts confirmed player candidates only.",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--min-color-share", type=float, default=MIN_COLOR_SHARE)
    parser.add_argument("--min-color-margin", type=float, default=MIN_COLOR_MARGIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def validate_args(args):
    if args.track_set != "confirmed":
        raise ValueError("Only --track-set confirmed is supported.")
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be a positive integer.")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be a positive integer.")
    for name in ("min_color_share", "min_color_margin"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if not args.input_tracks:
        raise ValueError("--input-tracks is required.")
    if not args.output_dir:
        raise ValueError("--output-dir is required.")


def upper_body_crop(frame, box):
    """Return a center torso crop, clipped to the image, or None if unusable."""
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = values
    height, width = frame.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 1 or box_height <= 1:
        return None
    left = max(0, int(math.floor(x1 + box_width * 0.22)))
    right = min(width, int(math.ceil(x1 + box_width * 0.78)))
    top = max(0, int(math.floor(y1 + box_height * 0.18)))
    bottom = min(height, int(math.ceil(y1 + box_height * 0.55)))
    if right - left < 2 or bottom - top < 2:
        return None
    crop = frame[top:bottom, left:right]
    return crop if crop.size else None


def upper_body_color_features(frame, box):
    crop = upper_body_crop(frame, box)
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    eligible = (saturation >= HSV_LOW_SATURATION) & (value >= HSV_LOW_VALUE)
    count = int(eligible.sum())
    if count < 12:
        return None
    red = ((hue <= 12) | (hue >= 170)) & eligible
    blue = (hue >= 92) & (hue <= 137) & eligible
    yellow = (hue >= 17) & (hue <= 32) & eligible
    green = (hue >= 33) & (hue <= 88) & eligible
    return {
        "team_a_red_share": float(red.sum() / count),
        "team_b_blue_share": float(blue.sum() / count),
        "referee_yellow_share": float(yellow.sum() / count),
        "goalkeeper_green_share": float(green.sum() / count),
    }


def _row_box(row):
    try:
        values = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Tracking CSV contains an invalid bounding box.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Tracking CSV contains a nonfinite bounding box.")
    return values


def aggregate_track_features(samples):
    aggregated = {}
    for key, feature_rows in samples.items():
        if not feature_rows:
            continue
        aggregated[key] = {
            name: float(np.median([row[name] for row in feature_rows]))
            for name in FEATURE_NAMES
        }
    return aggregated


def classify_track(features, sample_count, min_samples, min_share, min_margin):
    if sample_count < min_samples or not features:
        return "unknown", "unknown", 0.0
    ranked = sorted(features.items(), key=lambda item: item[1], reverse=True)
    best_name, best_share = ranked[0]
    second_share = ranked[1][1]
    if best_name == "team_a_red_share":
        if best_share < min_share or best_share - second_share < min_margin:
            return "unknown", "unknown", 0.0
        confidence = float(min(1.0, best_share * min(1.0, sample_count / 6.0)))
        return "team_a", "team_a_outfield", confidence
    if best_name == "team_b_blue_share":
        if best_share < min_share or best_share - second_share < min_margin:
            return "unknown", "unknown", 0.0
        confidence = float(min(1.0, best_share * min(1.0, sample_count / 6.0)))
        return "team_b", "team_b_outfield", confidence
    if best_name == "referee_yellow_share":
        if (
            sample_count < max(min_samples, MIN_SPECIAL_ROLE_SAMPLES)
            or best_share < 0.45
            or best_share - second_share < 0.05
        ):
            return "unknown", "unknown", 0.0
        confidence = float(min(1.0, best_share * min(1.0, sample_count / 6.0)))
        return "unknown", "referee_candidate", confidence
    if best_name == "goalkeeper_green_share":
        if (
            sample_count < max(min_samples, MIN_SPECIAL_ROLE_SAMPLES)
            or best_share < 0.65
            or best_share - second_share < 0.15
        ):
            return "unknown", "unknown", 0.0
        confidence = float(min(1.0, best_share * min(1.0, sample_count / 6.0)))
        return "unknown", "goalkeeper_candidate", confidence
    return "unknown", "unknown", 0.0


def _load_tracking_provenance(tracks_path, video_override=None):
    summary_path = tracks_path.with_name("tracking_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Tracking summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if tracks_path.name.lower() != "tracks_confirmed.csv":
        raise ValueError("Team classification requires tracks_confirmed.csv.")
    if summary.get("confirmed_candidates_csv") != "tracks_confirmed.csv":
        raise ValueError("Tracking summary does not identify confirmed candidates.")
    run_id = summary.get("tracking_run_id")
    if not run_id:
        raise ValueError("Tracking summary has no tracking_run_id.")
    recorded_video = summary.get("input_video")
    if not recorded_video:
        raise ValueError("Tracking summary has no input_video.")
    video_path = (
        video_override.expanduser().resolve()
        if video_override is not None
        else Path(recorded_video).expanduser().resolve()
    )
    if video_path != Path(recorded_video).expanduser().resolve():
        raise ValueError("--video does not match the video recorded in tracking summary.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    try:
        start_frame = int(summary["start_frame"])
        end_frame = int(summary["end_frame_exclusive"])
        fps = float(summary["video_fps"])
        width = int(summary["frame_width"])
        height = int(summary["frame_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tracking summary has invalid video/frame metadata.") from exc
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


def read_tracking_rows(path, run_id, start_frame, end_frame, fps):
    required = {
        "run_id", "frame", "timestamp", "track_id", "class_id",
        "x1", "y1", "x2", "y2",
    }
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required tracking columns.")
        previous_frame = -1
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != run_id:
                raise ValueError(f"Tracking run ID mismatch at CSV row {row_number}.")
            try:
                frame = int(row["frame"])
                timestamp = float(row["timestamp"])
                class_id = int(row["class_id"])
                track_id = str(row["track_id"]).strip()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid tracking data at CSV row {row_number}.") from exc
            if not start_frame <= frame < end_frame:
                raise ValueError(f"Frame {frame} is outside the tracking run range.")
            if frame < previous_frame:
                raise ValueError("Tracking CSV must be ordered by original frame.")
            if not math.isfinite(timestamp) or abs(timestamp - frame / fps) > 1e-3:
                raise ValueError(f"Invalid timestamp at tracking CSV row {row_number}.")
            _row_box(row)
            if class_id == PERSON_CLASS_ID and not track_id:
                raise ValueError(f"Person row {row_number} has no track ID.")
            row["frame"] = frame
            row["timestamp"] = timestamp
            row["class_id"] = class_id
            rows.append(row)
            previous_frame = frame
    return rows


def _classify_tracks(rows, frame_cache, sample_every, min_samples, min_share, min_margin):
    return _classify_tracks_with_frame_loader(
        rows,
        frame_cache.get,
        sample_every,
        min_samples,
        min_share,
        min_margin,
    )


def _classify_tracks_with_frame_loader(
    rows, frame_loader, sample_every, min_samples, min_share, min_margin
):
    samples = defaultdict(list)
    sample_rows_by_frame = defaultdict(list)
    for row in rows:
        if row["class_id"] != PERSON_CLASS_ID:
            continue
        frame_number = row["frame"]
        if frame_number % sample_every != 0:
            continue
        sample_rows_by_frame[frame_number].append(row)

    for frame_number in tqdm(
        sorted(sample_rows_by_frame), desc="Sampling jersey colors", unit="frame"
    ):
        frame = frame_loader(frame_number)
        if frame is None:
            raise ValueError(f"Missing source frame {frame_number} for color sampling.")
        for row in sample_rows_by_frame[frame_number]:
            features = upper_body_color_features(frame, _row_box(row))
            if features is not None:
                samples[(row["run_id"], row["track_id"])].append(features)
    aggregated = aggregate_track_features(samples)
    assignments = {}
    person_keys = {
        (row["run_id"], row["track_id"])
        for row in rows
        if row["class_id"] == PERSON_CLASS_ID and row["track_id"]
    }
    for key in sorted(person_keys, key=lambda item: (item[0], int(item[1]))):
        track_samples = samples.get(key, [])
        team, role, confidence = classify_track(
            aggregated.get(key, {}),
            len(track_samples),
            min_samples,
            min_share,
            min_margin,
        )
        assignments[key] = {
            "run_id": key[0],
            "track_id": key[1],
            "predicted_team": team,
            "predicted_role": role,
            "classification_confidence": confidence,
            "number_of_samples": len(track_samples),
            "classification_method": CLASSIFICATION_METHOD,
        }
    return assignments


def _role_color(role):
    return TEAM_COLORS.get(role, TEAM_COLORS["unknown"])


def _annotate_frame(frame, frame_rows, assignments):
    annotated = frame.copy()
    for row in frame_rows:
        if row["class_id"] != PERSON_CLASS_ID:
            continue
        key = (row["run_id"], row["track_id"])
        assignment = assignments.get(key)
        role = assignment["predicted_role"] if assignment else "unknown"
        color = _role_color(role)
        x1, y1, x2, y2 = [int(round(v)) for v in _row_box(row)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        confidence = assignment["classification_confidence"] if assignment else 0.0
        label = (
            f"ID {row['track_id']} | {DISPLAY_NAMES[role]} {confidence:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _write_outputs(
    output_dir, rows, assignments, video_path, start_frame, end_frame, fps,
    width, height, sample_every, run_id, summary,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / name for name in OUTPUT_NAMES}
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing team-classification output(s): "
            + ", ".join(str(path) for path in collisions)
        )

    rows_by_frame = defaultdict(list)
    for row in rows:
        rows_by_frame[row["frame"]].append(row)

    temp_paths = {}
    try:
        for name in OUTPUT_NAMES:
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{name}.",
                suffix=Path(name).suffix,
                dir=output_dir,
                delete=False,
            )
            temp_paths[name] = Path(handle.name)
            handle.close()

        with temp_paths["team_assignments.csv"].open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=ASSIGNMENT_FIELDS)
            writer.writeheader()
            for key in sorted(assignments, key=lambda item: (item[0], int(item[1]))):
                writer.writerow(assignments[key])

        summary_payload = {
            "classification_run_id": uuid4().hex,
            "tracking_run_id": run_id,
            "tracking_csv": str(summary["tracking_csv"]),
            "track_set": "confirmed",
            "input_video": str(video_path),
            "sampling": {
                "sample_every_n_frames": sample_every,
                "minimum_samples": summary["minimum_samples"],
                "minimum_color_share": summary["minimum_color_share"],
                "minimum_color_margin": summary["minimum_color_margin"],
                "minimum_special_role_samples": MIN_SPECIAL_ROLE_SAMPLES,
                "minimum_referee_yellow_share": 0.45,
                "minimum_referee_yellow_margin": 0.05,
                "minimum_goalkeeper_green_share": 0.65,
                "minimum_goalkeeper_green_margin": 0.15,
            },
            "classification_method": CLASSIFICATION_METHOD,
            "classification_is_provisional": True,
            "known_video_kit_description": {
                "team_a": "Burgundy/red outfield kit",
                "team_b": "Blue/dark outfield kit",
                "club_names_assigned": False,
            },
            "identity_scope": "track_id is scoped by tracking_run_id",
            "track_counts_by_role": dict(sorted(Counter(
                item["predicted_role"] for item in assignments.values()
            ).items())),
            "track_counts_by_team": dict(sorted(Counter(
                item["predicted_team"] for item in assignments.values()
            ).items())),
            "tracks": len(assignments),
            "source_person_observations": summary["source_person_observations"],
            "video_fps": fps,
            "frame_width": width,
            "frame_height": height,
            "tracking_frame_range": [start_frame, end_frame],
            "role_caveat": (
                "Goalkeeper and referee labels are color-based candidates, not "
                "verified roles. Unknown is used when evidence is insufficient. "
                "Color labels are not guaranteed to represent real-world identity."
            ),
            "outputs": list(OUTPUT_NAMES),
        }
        with temp_paths["team_classification_summary.json"].open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary_payload, file, indent=2)
            file.write("\n")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source video: {video_path}")
        writer = None
        preview_frames = []
        preview_targets = {
            start_frame,
            start_frame + (end_frame - start_frame - 1) // 4,
            start_frame + (end_frame - start_frame - 1) // 2,
            start_frame + 3 * (end_frame - start_frame - 1) // 4,
            end_frame - 1,
        }
        try:
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (
                not math.isfinite(actual_fps)
                or abs(actual_fps - fps) > 1e-3
                or actual_width != width
                or actual_height != height
                or end_frame > actual_count
            ):
                raise ValueError("Video metadata does not match tracking summary.")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                raise RuntimeError(f"Cannot seek to source frame {start_frame}.")
            writer = cv2.VideoWriter(
                str(temp_paths["annotated_teams.mp4"]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not initialize annotated video writer.")
            progress = tqdm(
                total=end_frame - start_frame,
                desc="Annotating team classifications",
                unit="frame",
            )
            try:
                for frame_number in range(start_frame, end_frame):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Could not read source video frame {frame_number}."
                        )
                    annotated = _annotate_frame(
                        frame, rows_by_frame.get(frame_number, []), assignments
                    )
                    writer.write(annotated)
                    if frame_number in preview_targets:
                        preview_frames.append(
                            cv2.resize(annotated, (640, 360))
                        )
                    progress.update(1)
            finally:
                progress.close()
        finally:
            if writer is not None:
                writer.release()
            capture.release()
        if not preview_frames:
            raise RuntimeError("No frames were available for the preview.")
        preview = cv2.vconcat(preview_frames)
        if not cv2.imwrite(
            str(temp_paths["team_classification_preview.png"]), preview
        ):
            raise RuntimeError("Could not write team-classification preview.")

        for name in OUTPUT_NAMES:
            if outputs[name].exists():
                raise FileExistsError(
                    f"Refusing to overwrite team output: {outputs[name]}"
                )
        for name in OUTPUT_NAMES:
            temp_paths[name].replace(outputs[name])
        return summary_payload
    except Exception:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise


def classify(args):
    validate_args(args)
    tracks_path = args.input_tracks.expanduser().resolve()
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {tracks_path}")
    (
        tracking_summary, run_id, video_path, start_frame, end_frame, fps,
        width, height,
    ) = _load_tracking_provenance(tracks_path, args.video)
    rows = read_tracking_rows(tracks_path, run_id, start_frame, end_frame, fps)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame > frame_count:
            raise ValueError("Tracking frame range exceeds source video length.")

        def load_sample_frame(frame_number):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
                raise RuntimeError(f"Cannot seek to source frame {frame_number}.")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read source frame {frame_number}.")
            return frame

        assignments = _classify_tracks_with_frame_loader(
            rows,
            load_sample_frame,
            args.sample_every,
            args.min_samples,
            args.min_color_share,
            args.min_color_margin,
        )
    finally:
        capture.release()
    summary = {
        "tracking_csv": str(tracks_path),
        "minimum_samples": args.min_samples,
        "minimum_color_share": args.min_color_share,
        "minimum_color_margin": args.min_color_margin,
        "source_person_observations": sum(
            row["class_id"] == PERSON_CLASS_ID for row in rows
        ),
    }
    return _write_outputs(
        args.output_dir.expanduser().resolve(),
        rows,
        assignments,
        video_path,
        start_frame,
        end_frame,
        fps,
        width,
        height,
        args.sample_every,
        run_id,
        summary,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = classify(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        csv.Error,
    ) as exc:
        parser.error(str(exc))
    print(f"Classified tracks: {summary['tracks']}")
    print(f"Role counts: {summary['track_counts_by_role']}")
    for filename in OUTPUT_NAMES:
        print(f"Saved: {args.output_dir / filename}")


if __name__ == "__main__":
    main()
