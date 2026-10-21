import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

import calibrate_pitch as pitch
import compensate_player_tracks as compensation


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data" / "input" / "sample.mp4"
DEFAULT_TRANSFORMS = ROOT / "data" / "output" / "camera_transforms_hybrid.npy"
DEFAULT_QUALITY = ROOT / "data" / "output" / "camera_quality_hybrid.csv"
DEFAULT_VALIDATION_SUMMARY = (
    ROOT / "data" / "output" / "landmark_validation" / "validation_summary.csv"
)
DEFAULT_CALIBRATION = (
    ROOT / "data" / "output" / "pitch_mapping" /
    "calibration_segment_000_run1" / "pitch_calibration.json"
)
SURVEY_FRAMES = (0, 50, 100, 150, 200, 250, 300, 350)
MAX_TRANSFORM_CONDITION = 1e12
DEFAULT_MAX_VALIDATION_ERROR_PX = 30.0
DEFAULT_MAX_REPEAT_ERROR_PX = 30.0
OBSERVATION_FIELDS = [
    "observation_id",
    "run_id",
    "camera_segment_id",
    "landmark_id",
    "landmark_name",
    "source_frame",
    "raw_image_x",
    "raw_image_y",
    "reference_frame",
    "reference_x",
    "reference_y",
    "transform_chain_frames",
    "transform_chain_condition",
    "transform_sha256",
    "quality_sha256",
    "source_video_sha256",
]
CORRESPONDENCE_FIELDS = pitch.CALIBRATION_FIELDS

VISIBILITY_CANDIDATES = [
    {
        "landmark_id": "L14",
        "landmark_name": "penalty_spot_left",
        "source_frames": "50;150;200;300",
        "assessment": "possible",
        "ambiguity": (
            "Small spot can be obscured by players, shadows, or motion blur; "
            "not consistently distinguishable in the reviewed frames."
        ),
    },
    {
        "landmark_id": "L17",
        "landmark_name": "goal_post_left_sideline_a",
        "source_frames": "50;150;200;300;350",
        "assessment": "possible_but_not_trusted",
        "ambiguity": (
            "The goal frame is visible in several shots, but the exact ground-plane "
            "contact point is blurred, partly clipped, or obscured by net/players."
        ),
    },
    {
        "landmark_id": "L18",
        "landmark_name": "goal_post_left_sideline_b",
        "source_frames": "50;150;200;300;350",
        "assessment": "possible_but_not_trusted",
        "ambiguity": (
            "The goal frame is visible in several shots, but the exact ground-plane "
            "contact point is blurred, partly clipped, or obscured by net/players."
        ),
    },
    {
        "landmark_id": "L10",
        "landmark_name": "penalty_area_left_sideline_a_corner",
        "source_frames": "50;150;200;300",
        "assessment": "not_confirmed",
        "ambiguity": (
            "The corresponding outer-box corner appears clipped or visually "
            "ambiguous; do not select a point merely along the white line."
        ),
    },
    {
        "landmark_id": "",
        "landmark_name": "halfway line, center circle, far-side penalty landmarks",
        "source_frames": "0;50;100;150;200;250;300;350",
        "assessment": "not_identifiably_visible",
        "ambiguity": (
            "The contact sheet does not show a uniquely identifiable center/halfway "
            "intersection or well-distributed opposite-end landmarks."
        ),
    },
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_validation_summary(path):
    best = {}
    path = Path(path)
    if not path.is_file():
        return best
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "test_frame", "created_at", "max_error_px", "median_error_px", "p90_error_px"
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing landmark-validation summary columns.")
        for row in reader:
            try:
                frame = int(row["test_frame"])
                maximum = float(row["max_error_px"])
                median = float(row["median_error_px"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid landmark-validation summary row.") from exc
            if not math.isfinite(maximum) or not math.isfinite(median):
                raise ValueError("Nonfinite landmark-validation error in summary.")
            candidate = {**row, "max_error_px": maximum, "median_error_px": median}
            if frame not in best or row["created_at"] > best[frame]["created_at"]:
                best[frame] = candidate
    return best


def compose_frame_chain(frame_transforms, frame, segment_id, reference_frame):
    if frame < 0 or frame >= len(frame_transforms):
        raise ValueError(f"Source frame {frame} is outside camera-transform coverage.")
    item = frame_transforms[frame]
    if item.segment_id != segment_id or item.reference_frame != reference_frame:
        raise ValueError(
            f"Frame {frame} is in {item.segment_id} referenced at "
            f"{item.reference_frame}, not requested segment {segment_id} "
            f"referenced at {reference_frame}."
        )
    if not item.valid or item.matrix is None:
        raise ValueError(
            f"Camera transform chain to frame {frame} is invalid: "
            f"{item.invalid_reason or 'invalid_transition'}."
        )
    matrix = np.asarray(item.matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Camera transform chain to frame {frame} is nonfinite.")
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > MAX_TRANSFORM_CONDITION:
        raise ValueError(
            f"Camera transform chain to frame {frame} is ill-conditioned "
            f"({condition:g})."
        )
    return matrix, condition


def validate_projection_domain(matrix, width, height):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Camera transform must be a finite 3x3 matrix.")
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > MAX_TRANSFORM_CONDITION:
        raise ValueError("Camera transform is singular or ill-conditioned.")
    denominators = [
        float(matrix[2] @ np.array([x, y, 1.0]))
        for x, y in ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
    ]
    if not np.isfinite(denominators).all():
        raise ValueError("Camera transform has nonfinite image-corner denominators.")
    scale = max(1.0, float(np.linalg.norm(matrix[2, :])))
    if min(denominators) <= 0 <= max(denominators):
        raise ValueError("Camera transform horizon crosses the selected frame.")
    if min(abs(value) for value in denominators) <= 1e-12 * scale:
        raise ValueError("Camera transform has a near-zero image-corner denominator.")
    return min(denominators), max(denominators)


def transform_landmark_point(matrix, x, y):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Landmark transform must be a finite 3x3 matrix.")
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > MAX_TRANSFORM_CONDITION:
        raise ValueError("Landmark transform is singular or ill-conditioned.")
    point, error = compensation.project_point(matrix, float(x), float(y))
    if error:
        raise ValueError(f"Landmark projection failed: {error}.")
    return point


def verify_observation_provenance(
    metadata,
    run_id,
    segment_id,
    reference_frame,
    source_video_sha256,
    transform_sha256,
    quality_sha256,
):
    expected = {
        "tracking_run_id": run_id,
        "camera_segment_id": segment_id,
        "reference_frame": int(reference_frame),
        "source_video_sha256": source_video_sha256,
        "camera_transform_sha256": transform_sha256,
        "camera_quality_sha256": quality_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Observation provenance mismatch for {key}.")


def theoretical_hull_scenarios(calibration):
    geometry = calibration["pitch_geometry"]
    hull = np.asarray(calibration["pitch_calibration_hull"], dtype=np.float32)
    landmarks = {
        row["landmark_id"]: (row["pitch_x_m"], row["pitch_y_m"])
        for row in pitch.named_landmarks(geometry)
    }
    baseline_area = float(cv2.contourArea(cv2.convexHull(hull)))
    pitch_area = geometry["length_m"] * geometry["width_m"]
    scenarios = {
        "penalty_spot_only": ["L14"],
        "goal_posts_only": ["L17", "L18"],
        "goal_posts_and_outer_box_corner": ["L10", "L14", "L17", "L18"],
    }
    result = {}
    for name, landmark_ids in scenarios.items():
        points = np.vstack((hull, np.asarray([landmarks[item] for item in landmark_ids])))
        area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
        result[name] = {
            "candidate_landmark_ids": landmark_ids,
            "canonical_pitch_hull_ratio": area / pitch_area,
            "gain_percentage_points_vs_baseline": 100 * (area - baseline_area) / pitch_area,
            "image_correspondences_assumed": False,
            "note": (
                "Canonical-coordinate geometry only; this is not evidence that "
                "the source-image landmarks are visible or usable."
            ),
        }
    return result


def group_landmark_observations(base_correspondences, observations):
    landmarks = {item["landmark_id"]: dict(item) for item in base_correspondences}
    grouped = defaultdict(list)
    for row in observations:
        landmark_id = row["landmark_id"]
        if landmark_id not in {item["landmark_id"] for item in pitch.named_landmarks()}:
            raise ValueError(f"Unknown canonical landmark ID: {landmark_id}.")
        if not all(
            math.isfinite(float(row[name]))
            for name in ("reference_x", "reference_y")
        ):
            raise ValueError(f"Nonfinite reference coordinate for {landmark_id}.")
        grouped[landmark_id].append(row)

    diagnostics = {}
    for landmark_id, rows in grouped.items():
        source_frames = [int(row["source_frame"]) for row in rows]
        if len(source_frames) != len(set(source_frames)):
            raise ValueError(
                f"Landmark {landmark_id} has multiple observations from one source frame."
            )
        points = np.asarray(
            [[float(row["reference_x"]), float(row["reference_y"])] for row in rows],
            dtype=np.float64,
        )
        median = np.median(points, axis=0)
        deviations = np.linalg.norm(points - median, axis=1)
        base = landmarks.get(landmark_id)
        if base is not None:
            base_point = np.array(
                [float(base["image_x"]), float(base["image_y"])], dtype=np.float64
            )
            deviations = np.append(deviations, np.linalg.norm(base_point - median))
        maximum = float(deviations.max()) if len(deviations) else 0.0
        canonical = next(
            item for item in pitch.named_landmarks()
            if item["landmark_id"] == landmark_id
        )
        landmarks[landmark_id] = {
            "correspondence_id": 0,
            "landmark_id": landmark_id,
            "landmark_name": canonical["landmark_name"],
            "reference_frame": int(rows[0]["reference_frame"]),
            "image_x": float(median[0]),
            "image_y": float(median[1]),
            "pitch_x_m": canonical["pitch_x_m"],
            "pitch_y_m": canonical["pitch_y_m"],
        }
        diagnostics[landmark_id] = {
            "observation_count": len(rows),
            "independent_pitch_location_count": 1,
            "reference_coordinate_median": median.tolist(),
            "maximum_repeat_deviation_px": maximum,
            "source_frames": sorted(set(source_frames)),
            "repeats_existing_calibration_landmark": base is not None,
        }
    result = [landmarks[key] for key in sorted(landmarks)]
    for index, row in enumerate(result, start=1):
        row["correspondence_id"] = index
    return result, diagnostics


def _read_base_correspondences(path, expected_run_id, segment_id, reference_frame):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = set(CORRESPONDENCE_FIELDS)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing calibration correspondence columns.")
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != expected_run_id:
                raise ValueError(f"Baseline run ID mismatch at row {row_number}.")
            if row["camera_segment_id"] != segment_id:
                raise ValueError(f"Baseline segment mismatch at row {row_number}.")
            if int(row["reference_frame"]) != reference_frame:
                raise ValueError(f"Baseline reference-frame mismatch at row {row_number}.")
            rows.append({
                **row,
                "correspondence_id": int(row["correspondence_id"]),
                "reference_frame": int(row["reference_frame"]),
                "image_x": float(row["image_x"]),
                "image_y": float(row["image_y"]),
                "pitch_x_m": float(row["pitch_x_m"]),
                "pitch_y_m": float(row["pitch_y_m"]),
            })
    if len({row["landmark_id"] for row in rows}) != len(rows):
        raise ValueError("Baseline calibration repeats a named landmark ID.")
    return rows


def _read_observations(
    directory,
    run_id,
    segment_id,
    reference_frame,
    source_video_sha256,
    transform_sha256,
    quality_sha256,
):
    directory = Path(directory)
    paths = sorted(directory.glob("observations_frame_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No multi-frame observations found under {directory}.")
    rows = []
    for path in paths:
        metadata_path = path.with_name(path.stem.replace("observations_", "provenance_") + ".json")
        metadata = _read_json(metadata_path, "Observation provenance")
        verify_observation_provenance(
            metadata,
            run_id,
            segment_id,
            reference_frame,
            source_video_sha256,
            transform_sha256,
            quality_sha256,
        )
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(OBSERVATION_FIELDS).issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing observation provenance columns.")
            for row_number, row in enumerate(reader, start=2):
                if (
                    row["run_id"] != run_id
                    or row["camera_segment_id"] != segment_id
                    or int(row["reference_frame"]) != reference_frame
                ):
                    raise ValueError(f"Observation run/segment mismatch at {path}:{row_number}.")
                if (
                    row["source_video_sha256"] != source_video_sha256
                    or row["transform_sha256"] != transform_sha256
                    or row["quality_sha256"] != quality_sha256
                ):
                    raise ValueError(f"Observation file provenance mismatch at {path}:{row_number}.")
                for key in (
                    "source_frame", "reference_frame", "raw_image_x", "raw_image_y",
                    "reference_x", "reference_y", "transform_chain_condition",
                ):
                    row[key] = float(row[key]) if key.endswith(("_x", "_y", "condition")) else int(row[key])
                rows.append(row)
    return rows


def _survey(args):
    video = args.video.expanduser().resolve()
    transforms_path = args.transforms.expanduser().resolve()
    quality_path = args.quality.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    calibration = _read_json(calibration_path, "Baseline calibration")
    if not calibration.get("valid_for_mapping"):
        raise ValueError("Baseline calibration is not valid for mapping.")
    if Path(calibration["source_video"]).resolve() != video:
        raise ValueError("Baseline calibration refers to another source video.")
    if _sha256(transforms_path) != calibration["camera_transform_sha256"]:
        raise ValueError("Camera transform file differs from baseline calibration.")
    if _sha256(quality_path) != calibration["camera_quality_sha256"]:
        raise ValueError("Camera quality file differs from baseline calibration.")
    transforms, quality = compensation.load_camera_data(transforms_path, quality_path)
    frame_transforms = compensation.build_frame_transforms(transforms, quality)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count != len(frame_transforms):
        capture.release()
        raise ValueError("Video and camera-transform frame counts differ.")
    if (
        [width, height] != calibration["source_video_dimensions"]
        or abs(fps - float(calibration["source_video_fps"])) > 1e-3
    ):
        capture.release()
        raise ValueError("Video metadata differs from baseline calibration.")
    thumb_w, thumb_h, label_h, columns = 480, 270, 36, 4
    sheet = np.zeros(
        (2 * (thumb_h + label_h), columns * thumb_w, 3), dtype=np.uint8
    )
    chains = []
    for index, frame_number in enumerate(SURVEY_FRAMES):
        if frame_number >= frame_count:
            capture.release()
            raise ValueError(f"Requested survey frame {frame_number} is unavailable.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Could not read survey frame {frame_number}.")
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        cv2.rectangle(sheet, (x, y), (x + thumb_w - 1, y + label_h - 1), (35, 35, 35), -1)
        cv2.putText(sheet, f"FRAME {frame_number}", (x + 12, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        sheet[y + label_h:y + label_h + thumb_h, x:x + thumb_w] = cv2.resize(
            frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA
        )
        item = frame_transforms[frame_number]
        record = {
            "frame": frame_number,
            "camera_segment_id": item.segment_id,
            "reference_frame": item.reference_frame,
            "valid": item.valid,
            "invalid_reason": item.invalid_reason,
        }
        if item.valid and item.matrix is not None:
            matrix = np.asarray(item.matrix, dtype=np.float64)
            denominator_min, denominator_max = validate_projection_domain(
                matrix, width, height
            )
            chain_steps = list(range(int(calibration["reference_frame"]) + 1, frame_number + 1))
            if any(
                not quality[step]["valid"] or quality[step]["method"] == "failed"
                for step in chain_steps
            ):
                raise ValueError(f"Frame {frame_number} crosses a failed camera transition.")
            record["condition_number"] = float(np.linalg.cond(matrix))
            record["matrix_finite"] = bool(np.isfinite(matrix).all())
            record["chain_transition_count"] = len(chain_steps)
            record["homogeneous_denominator_at_image_corners"] = [
                denominator_min, denominator_max
            ]
            record["transform_frame_to_reference"] = matrix.tolist()
        chains.append(record)
    capture.release()
    output = args.output_dir.expanduser().resolve()
    names = (
        "frame_contact_sheet_0_350.png",
        "camera_chain_inspection.json",
        "landmark_visibility_report.json",
        "landmark_visibility_report.csv",
        "landmark_visibility_report.md",
    )
    outputs, temporaries = pitch._atomic_output_paths(output, names)
    validation_by_frame = _read_validation_summary(args.validation_summary)
    report = {
        "source_video": str(video),
        "source_video_sha256": _sha256(video),
        "dimensions": [width, height],
        "fps": fps,
        "frame_count": frame_count,
        "inspected_frames": list(SURVEY_FRAMES),
        "tracking_run_id": calibration["tracking_run_id"],
        "camera_segment_id": calibration["camera_segment_id"],
        "reference_frame": calibration["reference_frame"],
        "candidate_landmarks": VISIBILITY_CANDIDATES,
        "theoretical_hull_scenarios": theoretical_hull_scenarios(calibration),
        "landmark_validation_by_frame": {
            str(frame): record for frame, record in sorted(
                validation_by_frame.items()
            ) if frame in SURVEY_FRAMES
        },
        "manual_validation_status_by_frame": {
            str(frame): (
                "validated"
                if frame in validation_by_frame
                else "not_manually_validated"
            )
            for frame in SURVEY_FRAMES
        },
        "conclusion": (
            "Possible additional observations are limited to the same goal-side "
            "penalty area. No uniquely identifiable halfway/center-circle or "
            "opposite-end landmarks were found in the inspected frames. Goal-post "
            "ground contacts and the penalty spot remain ambiguous and require "
            "manual confirmation; no candidate is accepted automatically."
        ),
    }
    report_csv_rows = [
        {key: row[key] for key in (
            "landmark_id", "landmark_name", "source_frames", "assessment", "ambiguity"
        )}
        for row in VISIBILITY_CANDIDATES
    ]
    markdown = [
        "# Pitch-landmark visibility survey",
        "",
        f"- Video: `{video}` ({width} x {height}, {fps:g} fps, {frame_count} frames)",
        f"- Inspected frames: {', '.join(map(str, SURVEY_FRAMES))}",
        f"- Camera segment: `{calibration['camera_segment_id']}`, reference "
        f"frame {calibration['reference_frame']}",
        "- Transform convention: `Hs[f]` maps frame f to frame f-1; "
        "`C[f] = C[f-1] @ Hs[f]` maps to the segment reference frame.",
        "",
        "| Landmark | Frames | Assessment | Ambiguity |",
        "|---|---|---|---|",
    ]
    for candidate in VISIBILITY_CANDIDATES:
        markdown.append(
            f"| {candidate['landmark_id']} {candidate['landmark_name']} | "
            f"{candidate['source_frames']} | {candidate['assessment']} | "
            f"{candidate['ambiguity']} |"
        )
    markdown.extend([
        "",
        "Canonical-only hull scenarios (no image coordinates inferred):",
        "",
        "| Scenario | Candidate IDs | Pitch hull coverage | Gain vs baseline |",
        "|---|---|---:|---:|",
    ])
    for name, scenario in report["theoretical_hull_scenarios"].items():
        markdown.append(
            f"| {name} | {', '.join(scenario['candidate_landmark_ids'])} | "
            f"{scenario['canonical_pitch_hull_ratio']:.2%} | "
            f"{scenario['gain_percentage_points_vs_baseline']:.2f} pp |"
        )
    markdown.extend([
        "",
        "These optimistic canonical-hull scenarios do not imply visible or accepted "
        "image correspondences.",
        "",
        report["conclusion"],
        "",
        "Camera-chain landmark validation is available at frames "
        + ", ".join(
            str(frame) for frame in SURVEY_FRAMES
            if str(frame) in report["landmark_validation_by_frame"]
        )
        + "; selected frames "
        + ", ".join(
            str(frame) for frame in SURVEY_FRAMES
            if str(frame) not in report["landmark_validation_by_frame"]
        )
        + " have no matching manual validation record.",
        "",
            "The contact sheet is a visual screening aid, not a calibration. Only "
        "manually verified, stationary, uniquely identifiable intersections "
        "should be selected.",
    ])
    committed = []
    try:
        if not cv2.imwrite(str(temporaries[names[0]]), sheet):
            raise RuntimeError("Could not save frame contact sheet.")
        temporaries[names[1]].write_text(
            json.dumps({
                "transform_convention": (
                    "Hs[f] maps frame f to frame f-1; C[f] = C[f-1] @ Hs[f]."
                ),
                "requested_frames": list(SURVEY_FRAMES),
                "camera_transform_sha256": _sha256(transforms_path),
                "camera_quality_sha256": _sha256(quality_path),
                "frame_chains": chains,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        temporaries[names[2]].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with temporaries[names[3]].open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report_csv_rows[0]))
            writer.writeheader()
            writer.writerows(report_csv_rows)
        temporaries[names[4]].write_text("\n".join(markdown) + "\n", encoding="utf-8")
        for name in names:
            if outputs[name].exists():
                raise FileExistsError(f"Refusing to overwrite {outputs[name]}")
        for name in names:
            temporaries[name].rename(outputs[name])
            committed.append(outputs[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        for path in committed:
            path.unlink(missing_ok=True)
        raise
    print(f"Contact sheet: {outputs[names[0]]}")
    print(f"Visibility report: {outputs[names[4]]}")
    return report


def _load_selection_context(args):
    video = args.video.expanduser().resolve()
    transforms_path = args.transforms.expanduser().resolve()
    quality_path = args.quality.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    calibration = _read_json(calibration_path, "Baseline calibration")
    if not calibration.get("valid_for_mapping"):
        raise ValueError("Baseline calibration is not valid for mapping.")
    if Path(calibration["source_video"]).resolve() != video:
        raise ValueError("Baseline calibration refers to a different video.")
    if _sha256(transforms_path) != calibration["camera_transform_sha256"]:
        raise ValueError("Camera transforms do not match baseline calibration provenance.")
    if _sha256(quality_path) != calibration["camera_quality_sha256"]:
        raise ValueError("Camera quality does not match baseline calibration provenance.")
    transforms, quality = compensation.load_camera_data(transforms_path, quality_path)
    frame_transforms = compensation.build_frame_transforms(transforms, quality)
    validation_by_frame = _read_validation_summary(args.validation_summary)
    matrix, condition = compose_frame_chain(
        frame_transforms,
        args.frame,
        calibration["camera_segment_id"],
        int(calibration["reference_frame"]),
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if args.frame >= count or len(frame_transforms) != count:
        capture.release()
        raise ValueError("Selected frame is outside video or transform coverage.")
    if (
        [width, height] != calibration["source_video_dimensions"]
        or abs(fps - float(calibration["source_video_fps"])) > 1e-3
    ):
        capture.release()
        raise ValueError("Video metadata differs from baseline calibration.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, source_frame = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(calibration["reference_frame"]))
    ref_ok, reference_frame = capture.read()
    capture.release()
    if not ok or source_frame is None or not ref_ok or reference_frame is None:
        raise RuntimeError("Could not read source or reference frame.")
    denominator_range = validate_projection_domain(matrix, width, height)
    return {
        "video": video,
        "transforms_path": transforms_path,
        "quality_path": quality_path,
        "calibration_path": calibration_path,
        "calibration": calibration,
        "matrix": matrix,
        "condition": condition,
        "denominator_range": denominator_range,
        "camera_validation": (
            validation_by_frame.get(args.frame)
            if args.frame != int(calibration["reference_frame"])
            else None
        ),
        "source_frame": source_frame,
        "reference_image": reference_frame,
        "width": width,
        "height": height,
    }


def _draw_selection_ui(source, reference, geometry, records, selected_id, context):
    panel_w, panel_h, header = 640, 360, 142
    canvas = np.zeros((header + panel_h, panel_w * 3, 3), dtype=np.uint8)
    src = cv2.resize(source, (panel_w, panel_h))
    ref = cv2.resize(reference, (panel_w, panel_h))
    pitch_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    rect = pitch.draw_pitch(pitch_panel, geometry)
    for row in records:
        source_point = (
            int(round(float(row["raw_image_x"]) * panel_w / source.shape[1])),
            int(round(float(row["raw_image_y"]) * panel_h / source.shape[0])),
        )
        ref_point = (
            int(round(float(row["reference_x"]) * panel_w / reference.shape[1])),
            int(round(float(row["reference_y"]) * panel_h / reference.shape[0])),
        )
        cv2.circle(src, source_point, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(src, row["landmark_id"], (source_point[0] + 5, source_point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(ref, ref_point, 6, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(ref, row["landmark_id"], (ref_point[0] + 5, ref_point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canonical = next(
            item for item in pitch.named_landmarks(geometry)
            if item["landmark_id"] == row["landmark_id"]
        )
        marker = pitch._pitch_to_panel(
            canonical["pitch_x_m"], canonical["pitch_y_m"], geometry, rect
        )
        cv2.circle(pitch_panel, marker, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(pitch_panel, row["landmark_id"], (marker[0] + 5, marker[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    landmarks = {item["landmark_id"]: item for item in pitch.named_landmarks(geometry)}
    for landmark_id, item in landmarks.items():
        point = pitch._pitch_to_panel(
            item["pitch_x_m"], item["pitch_y_m"], geometry, rect
        )
        color = (0, 255, 255) if landmark_id == selected_id else (240, 240, 240)
        cv2.circle(pitch_panel, point, 5, color, -1, cv2.LINE_AA)
        cv2.putText(pitch_panel, landmark_id, (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    canvas[header:, :panel_w] = src
    canvas[header:, panel_w:panel_w * 2] = ref
    canvas[header:, panel_w * 2:] = pitch_panel
    lines = [
        f"MULTI-FRAME LANDMARK | {context['calibration']['camera_segment_id']} | "
        f"source={context['frame']} reference={context['calibration']['reference_frame']}",
        "Left: source frame | middle: transformed point in reference frame | right: canonical pitch",
        "Click canonical landmark, then exact visible stationary intersection in source image.",
        "Do not use proximity alone, arbitrary line positions, players, or hidden points.",
        f"Completed observations: {len(records)} | selected: {selected_id or 'none'}",
        (
            f"Camera chain condition: {context['condition']:.3g} | "
            + (
                f"latest manual max error={context['camera_validation']['max_error_px']:.1f}px."
                if context["camera_validation"]
                else (
                    "reference frame; no camera transition."
                    if context["frame"] == context["calibration"]["reference_frame"]
                    else "no manual transition validation for this frame."
                )
            )
        ),
        "U: undo | R: reset | ENTER: save this frame | ESC: cancel",
    ]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 22 + index * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255) if index == 0 else (245, 245, 245),
                    1 if index else 2, cv2.LINE_AA)
    return canvas, rect, landmarks


def select_frame(args):
    context = _load_selection_context(args)
    geometry = pitch._geometry_from_args(args)
    records = []
    selected_id = None
    state = {"rect": None, "landmarks": {}}
    panel_w = 640

    def draw():
        canvas, rect, landmarks = _draw_selection_ui(
            context["source_frame"], context["reference_image"], geometry,
            records, selected_id, {**context, "frame": args.frame},
        )
        state["rect"] = rect
        state["landmarks"] = landmarks
        return canvas

    def callback(event, x, y, flags, userdata):
        nonlocal selected_id
        if event != cv2.EVENT_LBUTTONDOWN or y < 142 or y >= 502:
            return
        if x >= panel_w * 2:
            if selected_id is not None:
                print("Complete the image selection before choosing another landmark.")
                return
            px = x - panel_w * 2
            py = y - 142
            nearest = None
            distance = float("inf")
            for landmark_id, item in state["landmarks"].items():
                point = pitch._pitch_to_panel(
                    item["pitch_x_m"], item["pitch_y_m"], geometry, state["rect"]
                )
                current = math.hypot(point[0] - px, point[1] - py)
                if current < distance:
                    nearest, distance = landmark_id, current
            if nearest is None or distance > 18:
                print("Click a named canonical landmark marker.")
                return
            if any(row["landmark_id"] == nearest for row in records):
                print(f"{nearest} is already selected in this frame; use U to undo.")
                return
            selected_id = nearest
            return
        if x >= panel_w:
            print("Select a named pitch landmark, then click its point in the left source image.")
            return
        if selected_id is None:
            print("Choose the named canonical landmark on the right first.")
            return
        raw_x = x * context["width"] / panel_w
        raw_y = (y - 142) * context["height"] / 360
        ref_x, ref_y = transform_landmark_point(context["matrix"], raw_x, raw_y)
        records.append({
            "observation_id": len(records) + 1,
            "run_id": context["calibration"]["tracking_run_id"],
            "camera_segment_id": context["calibration"]["camera_segment_id"],
            "landmark_id": selected_id,
            "landmark_name": state["landmarks"][selected_id]["landmark_name"],
            "source_frame": args.frame,
            "raw_image_x": raw_x,
            "raw_image_y": raw_y,
            "reference_frame": int(context["calibration"]["reference_frame"]),
            "reference_x": ref_x,
            "reference_y": ref_y,
            "transform_chain_frames": json.dumps(
                list(range(int(context["calibration"]["reference_frame"]) + 1, args.frame + 1))
            ),
            "transform_chain_condition": context["condition"],
            "transform_sha256": _sha256(context["transforms_path"]),
            "quality_sha256": _sha256(context["quality_path"]),
            "source_video_sha256": _sha256(context["video"]),
        })
        print(f"Recorded {selected_id}: raw=({raw_x:.1f},{raw_y:.1f}) -> reference=({ref_x:.1f},{ref_y:.1f}).")
        selected_id = None

    window = f"Multi-frame landmark selection | frame {args.frame}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, callback)
    print("Select a named canonical landmark, then click its exact visible point in the source frame.")
    print("U = undo, R = reset, Enter = save, Esc = cancel.")
    try:
        while True:
            cv2.imshow(window, draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                if selected_id:
                    print("Complete or undo the pending landmark selection.")
                    continue
                if not records:
                    print("No complete landmark observations selected.")
                    continue
                break
            if key in (ord("u"), ord("U")):
                if selected_id:
                    selected_id = None
                elif records:
                    records.pop()
                print("Undid the latest selection.")
            elif key in (ord("r"), ord("R")):
                records.clear()
                selected_id = None
                print("Cleared selections for this frame.")
            elif key == 27:
                raise RuntimeError("Multi-frame landmark selection cancelled.")
    finally:
        cv2.destroyAllWindows()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{args.frame:04d}"
    csv_path = output / f"observations_{stem}.csv"
    provenance_path = output / f"provenance_{stem}.json"
    overlay_path = output / f"overlay_{stem}.png"
    for path in (csv_path, provenance_path, overlay_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    reference_overlay = context["reference_image"].copy()
    source_overlay = context["source_frame"].copy()
    for record in records:
        source_point = (int(round(record["raw_image_x"])), int(round(record["raw_image_y"])))
        reference_point = (int(round(record["reference_x"])), int(round(record["reference_y"])))
        cv2.circle(source_overlay, source_point, 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(source_overlay, record["landmark_id"], (source_point[0] + 8, source_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(reference_overlay, reference_point, 8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(reference_overlay, record["landmark_id"], (reference_point[0] + 8, reference_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    overlay = np.hstack((source_overlay, reference_overlay))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracking_run_id": context["calibration"]["tracking_run_id"],
        "camera_segment_id": context["calibration"]["camera_segment_id"],
        "reference_frame": int(context["calibration"]["reference_frame"]),
        "source_frame": args.frame,
        "source_video": str(context["video"]),
        "source_video_sha256": _sha256(context["video"]),
        "camera_transform_file": str(context["transforms_path"]),
        "camera_transform_sha256": _sha256(context["transforms_path"]),
        "camera_quality_file": str(context["quality_path"]),
        "camera_quality_sha256": _sha256(context["quality_path"]),
        "landmark_validation_summary": str(args.validation_summary.expanduser().resolve()),
        "landmark_validation_summary_sha256": (
            _sha256(args.validation_summary)
            if args.validation_summary.is_file()
            else None
        ),
        "camera_validation": (
            {
                "run_id": context["camera_validation"]["run_id"],
                "median_error_px": context["camera_validation"]["median_error_px"],
                "max_error_px": context["camera_validation"]["max_error_px"],
                "p90_error_px": context["camera_validation"]["p90_error_px"],
            }
            if context["camera_validation"]
            else None
        ),
        "composition_convention": "C[f] = C[f-1] @ Hs[f]",
        "transform_frame_to_reference": context["matrix"].tolist(),
        "transform_chain_condition": context["condition"],
        "transform_chain_frames": list(range(
            int(context["calibration"]["reference_frame"]) + 1, args.frame + 1
        )),
        "observation_count": len(records),
    }
    created_paths = []
    try:
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            created_paths.append(csv_path)
            writer = csv.DictWriter(stream, fieldnames=OBSERVATION_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        with provenance_path.open("x", encoding="utf-8") as stream:
            created_paths.append(provenance_path)
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        encoded, buffer = cv2.imencode(".png", overlay)
        if not encoded:
            raise RuntimeError("Could not encode transformed-point overlay.")
        with overlay_path.open("xb") as stream:
            created_paths.append(overlay_path)
            stream.write(buffer.tobytes())
    except Exception:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    print(f"Saved observations: {csv_path}")
    print(f"Transformed-point overlay: {overlay_path}")
    return records


def estimate_extended_calibration(args):
    baseline_path = args.calibration.expanduser().resolve()
    baseline = _read_json(baseline_path, "Baseline calibration")
    if not baseline.get("valid_for_mapping"):
        raise ValueError("Baseline calibration is not valid for mapping.")
    video = args.video.expanduser().resolve()
    if Path(baseline["source_video"]).resolve() != video:
        raise ValueError("Baseline calibration refers to a different source video.")
    if _sha256(args.transforms) != baseline["camera_transform_sha256"]:
        raise ValueError("Transform file differs from baseline calibration.")
    if _sha256(args.quality) != baseline["camera_quality_sha256"]:
        raise ValueError("Quality file differs from baseline calibration.")
    segment_id = baseline["camera_segment_id"]
    reference_frame = int(baseline["reference_frame"])
    run_id = baseline["tracking_run_id"]
    rows = _read_base_correspondences(
        baseline_path.with_name("calibration_correspondences.csv"),
        run_id,
        segment_id,
        reference_frame,
    )
    observations = _read_observations(
        args.observations_dir.expanduser().resolve(),
        run_id,
        segment_id,
        reference_frame,
        _sha256(video),
        _sha256(args.transforms),
        _sha256(args.quality),
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    transforms, quality = compensation.load_camera_data(args.transforms, args.quality)
    frame_transforms = compensation.build_frame_transforms(transforms, quality)
    if len(frame_transforms) != frame_count:
        capture.release()
        raise ValueError("Video and camera-transform frame counts differ.")
    for observation in observations:
        frame = int(observation["source_frame"])
        matrix, condition = compose_frame_chain(
            frame_transforms, frame, segment_id, reference_frame
        )
        if (
            observation["raw_image_x"] < 0
            or observation["raw_image_x"] >= width
            or observation["raw_image_y"] < 0
            or observation["raw_image_y"] >= height
        ):
            raise ValueError(f"Observation lies outside source frame {frame}.")
        expected_chain = list(range(reference_frame + 1, frame + 1))
        try:
            stored_chain = json.loads(observation["transform_chain_frames"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid transform-chain metadata for frame {frame}.") from exc
        if stored_chain != expected_chain:
            raise ValueError(f"Transform-chain frame list mismatch at frame {frame}.")
        if observation["transform_sha256"] != _sha256(args.transforms):
            raise ValueError(f"Transform provenance mismatch for source frame {frame}.")
        if observation["quality_sha256"] != _sha256(args.quality):
            raise ValueError(f"Quality provenance mismatch for source frame {frame}.")
        projected = transform_landmark_point(
            matrix, observation["raw_image_x"], observation["raw_image_y"]
        )
        if not np.allclose(
            projected,
            [observation["reference_x"], observation["reference_y"]],
            atol=0.25,
            rtol=0,
        ):
            raise ValueError(
                f"Stored reference coordinate does not match frame {frame} transform."
            )
        if abs(condition - observation["transform_chain_condition"]) > max(
            1e-6, condition * 1e-6
        ):
            raise ValueError(f"Transform-condition provenance mismatch at frame {frame}.")
    correspondences, repeat_diagnostics = group_landmark_observations(rows, observations)
    geometry = baseline["pitch_geometry"]
    image_points = [(row["image_x"], row["image_y"]) for row in correspondences]
    pitch_points = [(row["pitch_x_m"], row["pitch_y_m"]) for row in correspondences]
    ids = [row["landmark_id"] for row in correspondences]
    capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ok, reference_image = capture.read()
    capture.release()
    if not ok or reference_image is None:
        raise RuntimeError("Could not read baseline reference frame.")
    estimation = pitch.estimate_homography(
        ids, image_points, pitch_points, width, height, geometry,
        args.ransac_threshold_m,
    )
    validation = _read_validation_summary(args.validation_summary)
    frame_uncertainty = {}
    unknown_frames = []
    high_error_frames = []
    for frame in sorted({int(row["source_frame"]) for row in observations}):
        record = validation.get(frame)
        if frame == reference_frame:
            frame_uncertainty[str(frame)] = {
                "status": "reference_frame_no_camera_transform",
                "median_error_px": 0.0,
                "maximum_error_px": 0.0,
            }
        elif record is None:
            unknown_frames.append(frame)
            frame_uncertainty[str(frame)] = {"status": "no_manual_validation"}
        else:
            item = {
                "status": "validated",
                "run_id": record.get("run_id"),
                "median_error_px": record["median_error_px"],
                "maximum_error_px": record["max_error_px"],
                "p90_error_px": float(record["p90_error_px"]),
            }
            frame_uncertainty[str(frame)] = item
            if item["maximum_error_px"] > args.max_validation_error_px:
                high_error_frames.append(frame)
    inconsistent = {
        key: item["maximum_repeat_deviation_px"]
        for key, item in repeat_diagnostics.items()
        if item["maximum_repeat_deviation_px"] > args.max_repeat_error_px
    }
    baseline_pitch_ratio = float(
        baseline["estimation_diagnostics"]["pitch_hull_ratio"]
    )
    estimated_pitch_ratio = float(estimation.get("pitch_hull_ratio") or 0.0)
    unique_added = sorted(
        set(ids) - {row["landmark_id"] for row in rows}
    )
    if unknown_frames:
        estimation["errors"].append("camera_uncertainty_unvalidated_for_source_frames")
    if high_error_frames:
        estimation["errors"].append("camera_validation_error_exceeds_limit")
    if inconsistent:
        estimation["errors"].append("repeated_landmark_observations_inconsistent")
    if not unique_added:
        estimation["errors"].append("no_new_independent_pitch_landmarks")
    if estimated_pitch_ratio <= baseline_pitch_ratio + 1e-4:
        estimation["errors"].append("calibration_hull_does_not_expand")
    canonical_by_id = {
        row["landmark_id"]: (row["pitch_x_m"], row["pitch_y_m"])
        for row in pitch.named_landmarks(geometry)
    }
    reprojection_by_frame = defaultdict(list)
    if estimation["homography"] is not None:
        for observation in observations:
            projected, error = pitch.project_homography(
                estimation["homography"],
                observation["reference_x"],
                observation["reference_y"],
            )
            if error:
                reprojection_by_frame[str(observation["source_frame"])].append(None)
            else:
                expected = canonical_by_id[observation["landmark_id"]]
                reprojection_by_frame[str(observation["source_frame"])].append(
                    float(np.linalg.norm(np.asarray(projected) - np.asarray(expected)))
                )
    reprojection_errors_by_source_frame = {}
    for frame, values in reprojection_by_frame.items():
        usable = [value for value in values if value is not None and math.isfinite(value)]
        reprojection_errors_by_source_frame[frame] = {
            "observation_count": len(values),
            "valid_projection_count": len(usable),
            "median_error_assumed_m": float(np.median(usable)) if usable else None,
            "maximum_error_assumed_m": float(np.max(usable)) if usable else None,
        }
    estimation["valid_for_mapping"] = (
        estimation["valid_for_mapping"] and not estimation["errors"]
    )
    if estimation["valid_for_mapping"]:
        estimation["status"] = "valid_extended_calibration"
    else:
        estimation["status"] = "extended_calibration_unreliable"
    comparison = {
        "baseline_calibration_id": baseline["calibration_id"],
        "baseline_status": baseline["status"],
        "baseline_correspondences": baseline["correspondence_count"],
        "baseline_pitch_hull_ratio": baseline_pitch_ratio,
        "extended_correspondences": len(correspondences),
        "new_independent_landmark_ids": unique_added,
        "repeat_diagnostics": repeat_diagnostics,
        "camera_uncertainty_by_source_frame": frame_uncertainty,
        "reprojection_error_by_source_frame": reprojection_errors_by_source_frame,
        "camera_validation_missing_frames": unknown_frames,
        "camera_validation_high_error_frames": high_error_frames,
        "max_camera_validation_error_px": args.max_validation_error_px,
        "max_repeat_error_px": args.max_repeat_error_px,
        "extended_pitch_hull_ratio": estimated_pitch_ratio,
        "pitch_hull_ratio_change": estimated_pitch_ratio - baseline_pitch_ratio,
        "extended_estimation": estimation,
        "mapping_generated": False,
        "mapping_note": (
            "A mapping run is not auto-generated. Run the existing map command "
            "only after reviewing this new calibration and correspondence output."
        ),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "pitch_calibration.json",
        "calibration_correspondences.csv",
        "calibration_diagnostics.png",
        "multi_frame_observations.csv",
        "multi_frame_estimation.json",
    )
    _assert_new_output_files(output_dir, names)
    outputs, temporaries = pitch._atomic_output_paths(output_dir, names)
    committed = []
    updated = dict(baseline)
    updated.update({
        "calibration_id": uuid4().hex,
        "status": estimation["status"],
        "valid_for_mapping": estimation["valid_for_mapping"],
        "homography_image_to_pitch": estimation["homography"],
        "correspondence_count": len(correspondences),
        "landmark_ids": ids,
        "image_calibration_hull": cv2.convexHull(
            np.asarray(image_points, dtype=np.float32)
        ).reshape(-1, 2).tolist(),
        "pitch_calibration_hull": cv2.convexHull(
            np.asarray(pitch_points, dtype=np.float32)
        ).reshape(-1, 2).tolist(),
        "estimation_diagnostics": {
            key: value for key, value in estimation.items() if key != "homography"
        },
        "multi_frame_provenance": comparison,
    })
    if not estimation["valid_for_mapping"]:
        updated["mapping_scope"] = (
            "Not valid for mapping: one or more cross-frame, distribution, or "
            "homography checks failed."
        )
    try:
        temporaries[names[0]].write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        with temporaries[names[1]].open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CORRESPONDENCE_FIELDS)
            writer.writeheader()
            writer.writerows(correspondences)
        diagnostic = pitch.draw_calibration_diagnostic(
            reference_image,
            geometry,
            segment_id,
            reference_frame,
            "Extended calibration valid for mapping."
            if estimation["valid_for_mapping"]
            else "Extended calibration not valid for mapping; review diagnostics.",
            correspondences,
            estimation,
        )
        if not cv2.imwrite(str(temporaries[names[2]]), diagnostic):
            raise RuntimeError("Could not save extended calibration diagnostic.")
        with temporaries[names[3]].open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OBSERVATION_FIELDS)
            writer.writeheader()
            writer.writerows(observations)
        temporaries[names[4]].write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        for name in names:
            if outputs[name].exists():
                raise FileExistsError(f"Refusing to overwrite {outputs[name]}")
        for name in names:
            temporaries[name].rename(outputs[name])
            committed.append(outputs[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        for path in committed:
            path.unlink(missing_ok=True)
        raise
    print(f"Extended calibration: {outputs[names[0]]}")
    print(f"Status: {estimation['status']}")
    print(f"New independent landmarks: {unique_added}")
    print(f"Pitch hull coverage: {baseline_pitch_ratio:.4%} -> {estimated_pitch_ratio:.4%}")
    return comparison


def build_parser():
    parser = argparse.ArgumentParser(
        description="Survey and add manually verified landmarks across a single camera segment."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    survey = commands.add_parser("survey", help="Create frame contact sheet and chain report.")
    survey.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    survey.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    survey.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    survey.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    survey.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    survey.add_argument("--output-dir", type=Path, required=True)

    select = commands.add_parser("select", help="Select landmarks in one source frame.")
    select.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    select.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    select.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    select.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    select.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    select.add_argument("--frame", type=int, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    pitch._add_geometry_arguments(select)

    estimate = commands.add_parser("estimate", help="Re-estimate calibration without replacing baseline.")
    estimate.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    estimate.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    estimate.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    estimate.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    estimate.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    estimate.add_argument("--observations-dir", type=Path, required=True)
    estimate.add_argument("--output-dir", type=Path, required=True)
    estimate.add_argument("--ransac-threshold-m", type=float, default=pitch.RANSAC_REPROJECTION_THRESHOLD_M)
    estimate.add_argument("--max-validation-error-px", type=float, default=DEFAULT_MAX_VALIDATION_ERROR_PX)
    estimate.add_argument("--max-repeat-error-px", type=float, default=DEFAULT_MAX_REPEAT_ERROR_PX)
    return parser


def _assert_new_output_files(output_dir, names):
    conflicts = [Path(output_dir) / name for name in names if (Path(output_dir) / name).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite {conflicts[0]}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "survey":
            _survey(args)
        elif args.command == "select":
            if args.frame < 0:
                raise ValueError("--frame must be non-negative.")
            pitch._geometry_from_args(args)
            select_frame(args)
        else:
            for value, name in (
                (args.ransac_threshold_m, "--ransac-threshold-m"),
                (args.max_validation_error_px, "--max-validation-error-px"),
                (args.max_repeat_error_px, "--max-repeat-error-px"),
            ):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive.")
            estimate_extended_calibration(args)
    except (
        cv2.error,
        csv.Error,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
