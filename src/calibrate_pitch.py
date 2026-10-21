import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from tqdm import tqdm

import compensate_player_tracks as compensation


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data" / "input" / "sample.mp4"
DEFAULT_COMPENSATED = (
    ROOT / "data" / "output" / "compensated_tracking_run2"
    / "compensated_tracks.csv"
)
DEFAULT_TEAM_ASSIGNMENTS = (
    ROOT / "data" / "output" / "team_classification" / "team_assignments.csv"
)
DEFAULT_TRANSFORMS = ROOT / "data" / "output" / "camera_transforms_hybrid.npy"
DEFAULT_QUALITY = ROOT / "data" / "output" / "camera_quality_hybrid.csv"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "pitch_mapping"

DEFAULT_LENGTH_M = 105.0
DEFAULT_WIDTH_M = 68.0
DEFAULT_PENALTY_DEPTH_M = 16.5
DEFAULT_PENALTY_WIDTH_M = 40.32
DEFAULT_SIX_YARD_DEPTH_M = 5.5
DEFAULT_SIX_YARD_WIDTH_M = 18.32
DEFAULT_GOAL_WIDTH_M = 7.32
DEFAULT_PENALTY_SPOT_M = 11.0
DEFAULT_CENTER_CIRCLE_RADIUS_M = 9.15

MIN_CORRESPONDENCES = 4
MIN_TRAIN_CORRESPONDENCES = 4
RANSAC_REPROJECTION_THRESHOLD_M = 2.0
MAX_MEDIAN_REPROJECTION_ERROR_M = 2.5
MAX_HELD_OUT_ERROR_M = 6.0
MAX_CONDITION_NUMBER = 1e12
DENOMINATOR_EPSILON = 1e-12
MIN_SOURCE_HULL_RATIO = 1e-4
MIN_PITCH_HULL_RATIO = 0.01
MAX_IMAGE_WIDTH = 960
MAX_PANEL_HEIGHT = 540
CALIBRATION_PANEL_WIDTH = 960
CALIBRATION_HEADER_HEIGHT = 160
PITCH_BACKGROUND = (28, 92, 44)
PITCH_LINE = (230, 235, 225)
TEAM_COLORS = {
    "team_a_outfield": (38, 28, 150),
    "team_b_outfield": (220, 90, 30),
    "referee_candidate": (0, 220, 255),
    "goalkeeper_candidate": (40, 210, 40),
    "unknown": (160, 160, 160),
}
MAPPING_FIELDS = [
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
    "compensated_x",
    "compensated_y",
    "camera_segment_id",
    "reference_frame",
    "compensation_valid",
    "compensation_invalid_reason",
    "predicted_team",
    "predicted_role",
    "classification_confidence",
    "classification_samples",
    "classification_method",
    "pitch_x_m",
    "pitch_y_m",
    "pitch_mapping_valid",
    "pitch_mapping_invalid_reason",
]
CALIBRATION_FIELDS = [
    "correspondence_id",
    "run_id",
    "camera_segment_id",
    "landmark_id",
    "landmark_name",
    "reference_frame",
    "image_x",
    "image_y",
    "pitch_x_m",
    "pitch_y_m",
]
CALIBRATION_OUTPUTS = (
    "pitch_calibration.json",
    "calibration_correspondences.csv",
    "calibration_diagnostics.png",
)
MAPPING_OUTPUTS = (
    "mapped_player_positions.csv",
    "pitch_mapping_preview.png",
    "annotated_pitch_mapping.mp4",
    "mapping_summary.json",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Manually calibrate named pitch landmarks and map compensated "
            "player foot points into segment-local pitch coordinates."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for command_name in ("inspect", "calibrate"):
        subparser = commands.add_parser(
            command_name,
            help=(
                "Create a landmark-selection diagnostic."
                if command_name == "inspect"
                else "Open the named-landmark calibration interface."
            ),
        )
        _add_calibration_arguments(subparser)
        _add_geometry_arguments(subparser)
        subparser.add_argument("--output-dir", type=Path, default=None)
        if command_name == "calibrate":
            subparser.add_argument(
                "--ransac-threshold-m",
                type=float,
                default=RANSAC_REPROJECTION_THRESHOLD_M,
            )

    mapper = commands.add_parser(
        "map",
        help="Map a compensated tracking run using valid per-segment calibrations.",
    )
    mapper.add_argument("--input-compensated", type=Path, default=DEFAULT_COMPENSATED)
    mapper.add_argument(
        "--team-assignments", type=Path, default=DEFAULT_TEAM_ASSIGNMENTS
    )
    mapper.add_argument("--video", type=Path, default=None)
    mapper.add_argument("--calibration-dir", type=Path, required=True)
    mapper.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "mapped")
    _add_geometry_arguments(mapper)
    mapper.add_argument(
        "--calibration-boundary-tolerance-m",
        type=float,
        default=1.0,
        help="Allowed small pitch-boundary tolerance for projected feet.",
    )
    return parser


def _add_calibration_arguments(parser):
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--input-compensated", type=Path, default=DEFAULT_COMPENSATED)
    parser.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--segment-id", default="segment_000")


def _add_geometry_arguments(parser):
    parser.add_argument("--length-m", type=float, default=DEFAULT_LENGTH_M)
    parser.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M)
    parser.add_argument(
        "--penalty-area-depth-m", type=float, default=DEFAULT_PENALTY_DEPTH_M
    )
    parser.add_argument(
        "--penalty-area-width-m", type=float, default=DEFAULT_PENALTY_WIDTH_M
    )
    parser.add_argument(
        "--six-yard-box-depth-m", type=float, default=DEFAULT_SIX_YARD_DEPTH_M
    )
    parser.add_argument(
        "--six-yard-box-width-m", type=float, default=DEFAULT_SIX_YARD_WIDTH_M
    )
    parser.add_argument("--goal-width-m", type=float, default=DEFAULT_GOAL_WIDTH_M)
    parser.add_argument(
        "--penalty-spot-distance-m", type=float, default=DEFAULT_PENALTY_SPOT_M
    )
    parser.add_argument(
        "--center-circle-radius-m",
        type=float,
        default=DEFAULT_CENTER_CIRCLE_RADIUS_M,
    )


def _geometry_from_args(args):
    return pitch_geometry(
        length=args.length_m,
        width=args.width_m,
        penalty_depth=args.penalty_area_depth_m,
        penalty_width=args.penalty_area_width_m,
        six_yard_depth=args.six_yard_box_depth_m,
        six_yard_width=args.six_yard_box_width_m,
        goal_width=args.goal_width_m,
        penalty_spot=args.penalty_spot_distance_m,
        center_circle_radius=args.center_circle_radius_m,
    )


def pitch_geometry(
    length=DEFAULT_LENGTH_M,
    width=DEFAULT_WIDTH_M,
    penalty_depth=DEFAULT_PENALTY_DEPTH_M,
    penalty_width=DEFAULT_PENALTY_WIDTH_M,
    six_yard_depth=DEFAULT_SIX_YARD_DEPTH_M,
    six_yard_width=DEFAULT_SIX_YARD_WIDTH_M,
    goal_width=DEFAULT_GOAL_WIDTH_M,
    penalty_spot=DEFAULT_PENALTY_SPOT_M,
    center_circle_radius=DEFAULT_CENTER_CIRCLE_RADIUS_M,
):
    values = {
        "length_m": float(length),
        "width_m": float(width),
        "penalty_area_depth_m": float(penalty_depth),
        "penalty_area_width_m": float(penalty_width),
        "six_yard_box_depth_m": float(six_yard_depth),
        "six_yard_box_width_m": float(six_yard_width),
        "goal_width_m": float(goal_width),
        "penalty_spot_distance_m": float(penalty_spot),
        "center_circle_radius_m": float(center_circle_radius),
    }
    if not all(math.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError("All pitch dimensions must be finite and greater than zero.")
    if (
        penalty_depth >= length / 2
        or penalty_width >= width
        or six_yard_depth >= penalty_depth
        or six_yard_width >= penalty_width
        or goal_width >= width
        or penalty_spot >= penalty_depth
        or center_circle_radius >= min(length, width) / 2
    ):
        raise ValueError("Pitch geometry dimensions are inconsistent.")
    return values


def named_landmarks(geometry=None):
    geometry = geometry or pitch_geometry()
    length = geometry["length_m"]
    width = geometry["width_m"]
    pa_depth = geometry["penalty_area_depth_m"]
    pa_width = geometry["penalty_area_width_m"]
    six_depth = geometry["six_yard_box_depth_m"]
    six_width = geometry["six_yard_box_width_m"]
    goal_width = geometry["goal_width_m"]
    pa_top = (width - pa_width) / 2
    pa_bottom = (width + pa_width) / 2
    six_top = (width - six_width) / 2
    six_bottom = (width + six_width) / 2
    goal_top = (width - goal_width) / 2
    goal_bottom = (width + goal_width) / 2
    landmarks = []

    def add(name, x, y):
        landmarks.append({
            "landmark_id": f"L{len(landmarks) + 1:02d}",
            "landmark_name": name,
            "pitch_x_m": float(x),
            "pitch_y_m": float(y),
        })

    add("corner_goal_a_sideline_a", 0, 0)
    add("corner_goal_a_sideline_b", 0, width)
    add("corner_goal_b_sideline_a", length, 0)
    add("corner_goal_b_sideline_b", length, width)
    add("halfway_sideline_a", length / 2, 0)
    add("halfway_sideline_b", length / 2, width)
    add("center_spot", length / 2, width / 2)
    add("center_circle_sideline_a_intersection", length / 2, width / 2 - geometry["center_circle_radius_m"])
    add("center_circle_sideline_b_intersection", length / 2, width / 2 + geometry["center_circle_radius_m"])
    arc_radius = geometry["center_circle_radius_m"]
    arc_half_width = math.sqrt(
        max(
            0.0,
            arc_radius ** 2 - (pa_depth - geometry["penalty_spot_distance_m"]) ** 2,
        )
    )
    for side, x in (("left", 0.0), ("right", length)):
        direction = 1.0 if side == "left" else -1.0
        pa_x = direction * pa_depth if side == "left" else length - pa_depth
        six_x = direction * six_depth if side == "left" else length - six_depth
        add(f"penalty_area_{side}_sideline_a_corner", pa_x, pa_top)
        add(f"penalty_area_{side}_sideline_b_corner", pa_x, pa_bottom)
        add(f"six_yard_{side}_sideline_a_corner", six_x, six_top)
        add(f"six_yard_{side}_sideline_b_corner", six_x, six_bottom)
        spot_x = (
            geometry["penalty_spot_distance_m"]
            if side == "left"
            else length - geometry["penalty_spot_distance_m"]
        )
        add(f"penalty_spot_{side}", spot_x, width / 2)
        add(
            f"penalty_arc_{side}_sideline_a_intersection",
            pa_x,
            width / 2 - arc_half_width,
        )
        add(
            f"penalty_arc_{side}_sideline_b_intersection",
            pa_x,
            width / 2 + arc_half_width,
        )
        add(f"goal_post_{side}_sideline_a", x, goal_top)
        add(f"goal_post_{side}_sideline_b", x, goal_bottom)
    return landmarks


def _point_array(points, label):
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} points must have shape (n, 2).")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} points must be finite.")
    return array


def validate_correspondences(landmark_ids, image_points, pitch_points, width, height, geometry):
    image = _point_array(image_points, "Image")
    pitch = _point_array(pitch_points, "Pitch")
    if len(image) != len(pitch) or len(landmark_ids) != len(image):
        raise ValueError("Every image landmark must have exactly one pitch landmark.")
    if len(set(landmark_ids)) != len(landmark_ids):
        raise ValueError("A named pitch landmark may be selected only once.")
    if len(image) < MIN_CORRESPONDENCES:
        raise ValueError(
            f"At least {MIN_CORRESPONDENCES} complete correspondences are required."
        )
    if np.any(image[:, 0] < 0) or np.any(image[:, 0] >= width):
        raise ValueError("Image landmark x coordinates must lie inside the frame.")
    if np.any(image[:, 1] < 0) or np.any(image[:, 1] >= height):
        raise ValueError("Image landmark y coordinates must lie inside the frame.")
    if (
        np.any(pitch[:, 0] < 0)
        or np.any(pitch[:, 0] > geometry["length_m"])
        or np.any(pitch[:, 1] < 0)
        or np.any(pitch[:, 1] > geometry["width_m"])
    ):
        raise ValueError("Canonical landmark coordinates must lie on the pitch.")
    if np.linalg.matrix_rank(image - image.mean(axis=0)) < 2:
        raise ValueError("Image landmarks are collinear or geometrically degenerate.")
    if np.linalg.matrix_rank(pitch - pitch.mean(axis=0)) < 2:
        raise ValueError("Pitch landmarks are collinear or geometrically degenerate.")
    source_area = float(cv2.contourArea(cv2.convexHull(image.astype(np.float32))))
    pitch_area = float(cv2.contourArea(cv2.convexHull(pitch.astype(np.float32))))
    source_ratio = source_area / (float(width) * float(height))
    pitch_ratio = pitch_area / (geometry["length_m"] * geometry["width_m"])
    if source_ratio < MIN_SOURCE_HULL_RATIO:
        raise ValueError("Image landmark distribution covers too little image area.")
    if pitch_ratio < MIN_PITCH_HULL_RATIO:
        raise ValueError("Pitch landmarks are too tightly clustered for a stable map.")
    return {
        "source_hull_area_pixels": source_area,
        "source_hull_ratio": source_ratio,
        "pitch_hull_area_m2": pitch_area,
        "pitch_hull_ratio": pitch_ratio,
        "well_distributed": pitch_ratio >= 0.15 and source_ratio >= 0.01,
    }


def project_homography(matrix, x, y, denominator_epsilon=DENOMINATOR_EPSILON):
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None, "nonfinite_input_point"
    point = np.asarray(matrix, dtype=np.float64) @ np.array(
        [float(x), float(y), 1.0], dtype=np.float64
    )
    if not np.isfinite(point).all():
        return None, "nonfinite_projected_point"
    scale = max(1.0, float(np.linalg.norm(matrix[2, :])))
    if abs(float(point[2])) <= denominator_epsilon * scale:
        return None, "near_zero_homogeneous_denominator"
    projected = point[:2] / point[2]
    if not np.isfinite(projected).all():
        return None, "nonfinite_projected_coordinates"
    return (float(projected[0]), float(projected[1])), ""


def _homography_condition(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        return "invalid_homography_shape", None
    if not np.isfinite(matrix).all():
        return "nonfinite_homography", None
    norm = float(np.linalg.norm(matrix))
    if norm <= 0:
        return "singular_homography", None
    normalized = matrix / norm
    try:
        determinant = float(np.linalg.det(normalized))
        condition = float(np.linalg.cond(normalized))
    except np.linalg.LinAlgError:
        return "singular_homography", None
    if not math.isfinite(determinant) or abs(determinant) < 1e-14:
        return "singular_homography", condition
    if not math.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        return "ill_conditioned_homography", condition
    return "", condition


def _local_orientation(matrix, image_points):
    center = np.mean(image_points, axis=0)
    return _orientation_at(matrix, center[0], center[1])


def _orientation_at(matrix, x, y):
    h = matrix
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(denominator) <= DENOMINATOR_EPSILON:
        return None
    u_numerator = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v_numerator = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    denominator_squared = denominator * denominator
    du_dx = (h[0, 0] * denominator - u_numerator * h[2, 0]) / denominator_squared
    du_dy = (h[0, 1] * denominator - u_numerator * h[2, 1]) / denominator_squared
    dv_dx = (h[1, 0] * denominator - v_numerator * h[2, 0]) / denominator_squared
    dv_dy = (h[1, 1] * denominator - v_numerator * h[2, 1]) / denominator_squared
    return float(du_dx * dv_dy - du_dy * dv_dx)


def estimate_homography(
    landmark_ids,
    image_points,
    pitch_points,
    width,
    height,
    geometry,
    ransac_threshold_m=RANSAC_REPROJECTION_THRESHOLD_M,
):
    distribution = validate_correspondences(
        landmark_ids, image_points, pitch_points, width, height, geometry
    )
    if not math.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0:
        raise ValueError("--ransac-threshold-m must be finite and positive.")
    image = _point_array(image_points, "Image")
    pitch = _point_array(pitch_points, "Pitch")
    count = len(image)
    result = {
        "homography": None,
        "valid_for_mapping": False,
        "status": "unvalidated",
        "errors": [],
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_m": None,
        "maximum_reprojection_error_m": None,
        "held_out_landmark_id": None,
        "held_out_error_m": None,
        "orientation_jacobian_determinant": None,
        **distribution,
    }

    if count >= 5:
        train_image, train_pitch = image[:-1], pitch[:-1]
        held_image, held_pitch = image[-1:], pitch[-1:]
        if np.linalg.matrix_rank(train_image - train_image.mean(axis=0)) < 2:
            result["errors"].append("training_correspondences_collinear")
        else:
            train_matrix, _ = cv2.findHomography(
                train_image,
                train_pitch,
                cv2.RANSAC,
                ransac_threshold_m,
            )
            if train_matrix is not None and np.isfinite(train_matrix).all():
                held_projection, held_error = project_homography(
                    train_matrix, held_image[0, 0], held_image[0, 1]
                )
                if held_error:
                    result["errors"].append(f"held_out_{held_error}")
                else:
                    held_error_m = float(
                        np.linalg.norm(np.asarray(held_projection) - held_pitch[0])
                    )
                    result["held_out_error_m"] = held_error_m
                    result["held_out_landmark_id"] = landmark_ids[-1]
                    if held_error_m > MAX_HELD_OUT_ERROR_M:
                        result["errors"].append("held_out_error_exceeds_limit")
            else:
                result["errors"].append("held_out_training_homography_failed")
    else:
        result["errors"].append("held_out_validation_requires_at_least_five_points")

    matrix, inliers = cv2.findHomography(
        image,
        pitch,
        cv2.RANSAC,
        ransac_threshold_m,
    )
    if matrix is None:
        result["errors"].append("ransac_homography_estimation_failed")
        result["status"] = "invalid"
        return result
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix_error, condition = _homography_condition(matrix)
    result["condition_number"] = condition
    if matrix_error:
        result["errors"].append(matrix_error)
    if inliers is not None:
        inlier_mask = inliers.reshape(-1).astype(bool)
        result["inlier_count"] = int(inlier_mask.sum())
        result["inlier_ratio"] = float(inlier_mask.mean())
    else:
        inlier_mask = np.ones(count, dtype=bool)
        result["inlier_count"] = count
        result["inlier_ratio"] = 1.0
    errors = []
    for image_point, pitch_point in zip(image[inlier_mask], pitch[inlier_mask]):
        projected, error = project_homography(matrix, *image_point)
        if error:
            result["errors"].append(error)
            continue
        errors.append(float(np.linalg.norm(np.asarray(projected) - pitch_point)))
    if errors:
        result["median_reprojection_error_m"] = float(np.median(errors))
        result["maximum_reprojection_error_m"] = float(np.max(errors))
        if result["median_reprojection_error_m"] > MAX_MEDIAN_REPROJECTION_ERROR_M:
            result["errors"].append("median_reprojection_error_exceeds_limit")
    if result["inlier_count"] < MIN_TRAIN_CORRESPONDENCES:
        result["errors"].append("fewer_than_four_ransac_inliers")
    if result["inlier_ratio"] < 0.6:
        result["errors"].append("ransac_inlier_ratio_below_0_60")

    orientation = _local_orientation(matrix, image)
    result["orientation_jacobian_determinant"] = orientation
    orientation_values = [
        _orientation_at(matrix, x, y) for x, y in image
    ]
    if (
        orientation is None
        or not math.isfinite(orientation)
        or abs(orientation) <= 1e-12
        or any(value is None or not math.isfinite(value) for value in orientation_values)
        or any(value * orientation <= 0 for value in orientation_values)
    ):
        result["errors"].append("inconsistent_or_invalid_geographic_orientation")
        result["geographic_orientation"] = "invalid_or_inconsistent"
    else:
        result["geographic_orientation"] = (
            "positive_image_to_pitch" if orientation > 0
            else "negative_image_to_pitch"
        )
        result["orientation_warning"] = (
            ""
            if orientation > 0
            else (
                "Image-to-pitch axes have reversed local orientation. This may be "
                "correct for the camera viewpoint; verify named goal and sideline "
                "correspondences before accepting."
            )
        )
    # Reject a projective horizon crossing the source-frame rectangle.
    denominator_corners = [
        matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    if min(denominator_corners) <= 0 <= max(denominator_corners):
        result["errors"].append("homogeneous_horizon_crosses_source_frame")
    result["homography"] = matrix.tolist()
    if count < 5:
        result["status"] = "insufficient_independent_validation"
    elif result["errors"]:
        result["status"] = "unreliable"
    else:
        result["status"] = "valid"
        result["valid_for_mapping"] = True
    return result


def _read_json(path, required_name):
    if not path.is_file():
        raise FileNotFoundError(f"{required_name} not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compensation_context(compensated_path, video_override=None):
    summary_path = compensated_path.with_name("compensation_summary.json")
    summary = _read_json(summary_path, "Compensation summary")
    run_id = summary.get("tracking_run_id")
    video_name = summary.get("input_video")
    if not run_id or not video_name:
        raise ValueError("Compensation summary lacks run or source-video provenance.")
    recorded_video = Path(video_name).expanduser().resolve()
    video_path = video_override.expanduser().resolve() if video_override else recorded_video
    if video_path != recorded_video:
        raise ValueError("Selected video does not match compensation summary.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    return summary, run_id, video_path


def load_tracking_context(compensated_path, team_path, video_override=None):
    compensation_summary, run_id, video_path = load_compensation_context(
        compensated_path, video_override
    )
    # The tracking provenance is located beside the original confirmed CSV, which
    # may not be adjacent to the compensated output. The summary carries its path.
    tracking_csv = compensation_summary.get("tracking_csv")
    if not tracking_csv:
        raise ValueError("Compensation summary does not identify its tracking CSV.")
    tracking_summary_path = Path(tracking_csv).with_name("tracking_summary.json")
    tracking_summary = _read_json(tracking_summary_path, "Tracking summary")
    if tracking_summary.get("tracking_run_id") != run_id:
        raise ValueError("Compensation summary and tracking summary run IDs differ.")
    if Path(tracking_summary.get("input_video", "")).expanduser().resolve() != video_path:
        raise ValueError("Tracking and compensation summaries refer to different videos.")

    team_summary_path = team_path.with_name("team_classification_summary.json")
    team_summary = _read_json(team_summary_path, "Team-classification summary")
    if team_summary.get("tracking_run_id") != run_id:
        raise ValueError("Team assignments belong to a different tracking run.")
    if Path(team_summary.get("input_video", "")).expanduser().resolve() != video_path:
        raise ValueError("Team assignments refer to a different source video.")
    team_tracking_path = team_summary.get("tracking_csv")
    tracking_csv_path = compensation_summary.get("tracking_csv")
    if (
        not team_tracking_path
        or not tracking_csv_path
        or Path(team_tracking_path).expanduser().resolve()
        != Path(tracking_csv_path).expanduser().resolve()
    ):
        raise ValueError("Team assignments were produced from a different tracking CSV.")
    return compensation_summary, tracking_summary, team_summary, run_id, video_path


def read_compensated_rows(path, expected_run_id):
    required = {
        "run_id", "frame", "timestamp", "track_id", "class_id", "confidence",
        "x1", "y1", "x2", "y2", "foot_x", "foot_y", "compensated_x",
        "compensated_y", "camera_segment_id", "reference_frame",
        "compensation_valid", "invalid_reason",
    }
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing compensated tracking columns.")
        previous_frame = -1
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != expected_run_id:
                raise ValueError(f"Tracking run ID mismatch at row {row_number}.")
            try:
                row["frame"] = int(row["frame"])
                row["reference_frame"] = int(row["reference_frame"])
                row["class_id"] = int(row["class_id"])
                row["timestamp"] = float(row["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame metadata at row {row_number}.") from exc
            if row["frame"] < previous_frame:
                raise ValueError("Compensated tracking CSV must be frame ordered.")
            if not math.isfinite(row["timestamp"]):
                raise ValueError(f"Nonfinite timestamp at row {row_number}.")
            if row["class_id"] == 0 and not row["track_id"].strip():
                raise ValueError(f"Person row {row_number} has no track ID.")
            previous_frame = row["frame"]
            rows.append(row)
    return rows


def read_team_assignments(path, expected_run_id):
    required = {
        "run_id", "track_id", "predicted_team", "predicted_role",
        "classification_confidence", "number_of_samples", "classification_method",
    }
    assignments = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing team-assignment columns.")
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != expected_run_id:
                raise ValueError(f"Team assignment run ID mismatch at row {row_number}.")
            if not row["track_id"]:
                raise ValueError(f"Missing track ID in team assignment row {row_number}.")
            key = (row["run_id"], row["track_id"])
            if key in assignments:
                raise ValueError(f"Duplicate team assignment for run/track {key}.")
            try:
                confidence = float(row["classification_confidence"])
                samples = int(row["number_of_samples"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid team classification data at row {row_number}.") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1 or samples < 0:
                raise ValueError(f"Invalid team classification data at row {row_number}.")
            assignments[key] = row
    return assignments


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_output_paths(output_dir, names):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / name for name in names}
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing pitch-mapping output(s): "
            + ", ".join(map(str, collisions))
        )
    temporaries = {}
    for name in names:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{Path(name).stem}.",
            suffix=Path(name).suffix,
            dir=output_dir,
            delete=False,
        )
        temporaries[name] = Path(handle.name)
        handle.close()
    return outputs, temporaries


def _pitch_to_panel(x, y, geometry, rect):
    x0, y0, width, height = rect
    return (
        int(round(x0 + float(x) / geometry["length_m"] * width)),
        int(round(y0 + float(y) / geometry["width_m"] * height)),
    )


def draw_pitch(canvas, geometry, selected_ids=(), calibration=None):
    canvas[:] = (24, 48, 32)
    height, width = canvas.shape[:2]
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.12)
    pitch_width = width - 2 * margin_x
    pitch_height = int(pitch_width * geometry["width_m"] / geometry["length_m"])
    if pitch_height > height - 2 * margin_y:
        pitch_height = height - 2 * margin_y
        pitch_width = int(pitch_height * geometry["length_m"] / geometry["width_m"])
    x0 = (width - pitch_width) // 2
    y0 = max(44, (height - pitch_height) // 2)
    rect = (x0, y0, pitch_width, pitch_height)
    pitch_top_left = _pitch_to_panel(0, 0, geometry, rect)
    pitch_bottom_right = _pitch_to_panel(
        geometry["length_m"], geometry["width_m"], geometry, rect
    )
    cv2.rectangle(canvas, pitch_top_left, pitch_bottom_right, PITCH_BACKGROUND, -1)
    cv2.rectangle(canvas, pitch_top_left, pitch_bottom_right, PITCH_LINE, 2)

    def line(a, b, thickness=1):
        cv2.line(
            canvas,
            _pitch_to_panel(*a, geometry, rect),
            _pitch_to_panel(*b, geometry, rect),
            PITCH_LINE,
            thickness,
            cv2.LINE_AA,
        )

    length, pitch_width_m = geometry["length_m"], geometry["width_m"]
    center_x, center_y = length / 2, pitch_width_m / 2
    line((center_x, 0), (center_x, pitch_width_m))
    center_pixel = _pitch_to_panel(center_x, center_y, geometry, rect)
    circle_radius = int(
        round(geometry["center_circle_radius_m"] * pitch_width / length)
    )
    cv2.circle(canvas, center_pixel, circle_radius, PITCH_LINE, 1, cv2.LINE_AA)
    cv2.circle(canvas, center_pixel, 3, PITCH_LINE, -1, cv2.LINE_AA)
    for side in ("left", "right"):
        base_x = 0 if side == "left" else length
        pa_x = geometry["penalty_area_depth_m"] if side == "left" else length - geometry["penalty_area_depth_m"]
        six_x = geometry["six_yard_box_depth_m"] if side == "left" else length - geometry["six_yard_box_depth_m"]
        pa_top = (pitch_width_m - geometry["penalty_area_width_m"]) / 2
        pa_bottom = (pitch_width_m + geometry["penalty_area_width_m"]) / 2
        six_top = (pitch_width_m - geometry["six_yard_box_width_m"]) / 2
        six_bottom = (pitch_width_m + geometry["six_yard_box_width_m"]) / 2
        line((base_x, pa_top), (pa_x, pa_top))
        line((pa_x, pa_top), (pa_x, pa_bottom))
        line((pa_x, pa_bottom), (base_x, pa_bottom))
        line((base_x, six_top), (six_x, six_top))
        line((six_x, six_top), (six_x, six_bottom))
        line((six_x, six_bottom), (base_x, six_bottom))
        spot_x = geometry["penalty_spot_distance_m"] if side == "left" else length - geometry["penalty_spot_distance_m"]
        spot = _pitch_to_panel(spot_x, center_y, geometry, rect)
        cv2.circle(canvas, spot, 3, PITCH_LINE, -1, cv2.LINE_AA)
        radius = geometry["center_circle_radius_m"]
        offset = geometry["penalty_area_depth_m"] - geometry["penalty_spot_distance_m"]
        alpha = math.acos(offset / radius)
        if side == "left":
            angles = np.linspace(-alpha, alpha, 48)
        else:
            angles = np.linspace(math.pi - alpha, math.pi + alpha, 48)
        arc_points = [
            _pitch_to_panel(
                spot_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
                geometry,
                rect,
            )
            for angle in angles
        ]
        cv2.polylines(
            canvas,
            [np.asarray(arc_points, dtype=np.int32)],
            False,
            PITCH_LINE,
            1,
            cv2.LINE_AA,
        )
    if calibration and calibration.get("image_hull"):
        pitch_polygon = []
        for x, y in calibration["pitch_hull"]:
            pitch_polygon.append(_pitch_to_panel(x, y, geometry, rect))
        if len(pitch_polygon) >= 3:
            cv2.polylines(
                canvas,
                [np.asarray(pitch_polygon, dtype=np.int32)],
                True,
                (0, 210, 220),
                2,
                cv2.LINE_AA,
            )
    return rect


def draw_calibration_diagnostic(
    frame,
    geometry,
    segment_id,
    reference_frame,
    status,
    correspondences=(),
    estimation=None,
):
    height = CALIBRATION_HEADER_HEIGHT + MAX_PANEL_HEIGHT
    width = CALIBRATION_PANEL_WIDTH * 2
    left_width = width // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = cv2.resize(frame, (left_width, MAX_PANEL_HEIGHT))
    canvas[
        CALIBRATION_HEADER_HEIGHT:, :left_width
    ] = left
    right = canvas[CALIBRATION_HEADER_HEIGHT:, left_width:]
    draw_pitch(right, geometry)
    messages = [
        f"CAMERA SEGMENT {segment_id} | REFERENCE FRAME {reference_frame}",
        status,
        "Manual named-landmark correspondences required; no points are inferred.",
        "105 x 68 m is an assumed default, not verified stadium measurement.",
    ]
    if estimation:
        messages.append(
            f"Correspondences: {len(correspondences)} | status: {estimation['status']}"
        )
        if estimation["median_reprojection_error_m"] is not None:
            messages.append(
                "Median reprojection error: "
                f"{estimation['median_reprojection_error_m']:.2f} assumed m"
            )
        if estimation["held_out_error_m"] is not None:
            messages.append(
                f"Held-out error: {estimation['held_out_error_m']:.2f} assumed m"
            )
        for error in estimation["errors"][:3]:
            messages.append(f"Check: {error}")
        if estimation.get("orientation_warning"):
            messages.append(estimation["orientation_warning"])
    for index, message in enumerate(messages):
        y = 24 + index * 23
        color = (0, 255, 255) if index < 2 else (245, 245, 245)
        cv2.putText(
            canvas,
            message,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    for index, item in enumerate(correspondences):
        point = (
            int(round(float(item["image_x"]) * left_width / frame.shape[1])),
            CALIBRATION_HEADER_HEIGHT
            + int(round(float(item["image_y"]) * MAX_PANEL_HEIGHT / frame.shape[0])),
        )
        cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(item["landmark_id"]),
            (point[0] + 6, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _calibration_context(args):
    video_path = args.video.expanduser().resolve()
    compensated_path = args.input_compensated.expanduser().resolve()
    transforms_path = args.transforms.expanduser().resolve()
    quality_path = args.quality.expanduser().resolve()
    for path, name in (
        (video_path, "Video"),
        (compensated_path, "Compensated tracking CSV"),
        (transforms_path, "Camera transform array"),
        (quality_path, "Camera quality CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    summary, run_id, recorded_video = load_compensation_context(
        compensated_path, video_path
    )
    if recorded_video != video_path:
        raise ValueError("Calibration video differs from compensated tracking video.")
    transforms, quality = compensation.load_camera_data(transforms_path, quality_path)
    frame_transforms = compensation.build_frame_transforms(transforms, quality)
    segment_frames = [
        index for index, frame in enumerate(frame_transforms)
        if frame.segment_id == args.segment_id
    ]
    if not segment_frames:
        raise ValueError(f"Unknown camera segment: {args.segment_id}")
    reference_frame = frame_transforms[segment_frames[0]].reference_frame
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (
        not math.isfinite(fps)
        or abs(fps - float(summary["source_video_fps"])) > 1e-3
        or width != int(summary["source_frame_width"])
        or height != int(summary["source_frame_height"])
        or frame_count != int(summary["source_frame_count"])
    ):
        capture.release()
        raise ValueError("Video metadata differs from compensation summary.")
    if len(frame_transforms) != frame_count:
        capture.release()
        raise ValueError("Camera transform coverage differs from source video frames.")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame):
        capture.release()
        raise RuntimeError(f"Could not seek to camera reference frame {reference_frame}.")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read camera reference frame {reference_frame}.")
    return {
        "video_path": video_path,
        "compensated_path": compensated_path,
        "tracking_run_id": run_id,
        "compensation_summary": summary,
        "transforms_path": transforms_path,
        "quality_path": quality_path,
        "frame_transforms": frame_transforms,
        "segment_id": args.segment_id,
        "reference_frame": reference_frame,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frame": frame,
    }


def _draw_calibration_ui(
    frame, geometry, correspondences, selected_id, segment_id, reference_frame
):
    panel_height = CALIBRATION_HEADER_HEIGHT + MAX_PANEL_HEIGHT
    panel_width = CALIBRATION_PANEL_WIDTH * 2
    left_width = panel_width // 2
    canvas = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    image_panel = cv2.resize(frame, (left_width, MAX_PANEL_HEIGHT))
    canvas[
        CALIBRATION_HEADER_HEIGHT:CALIBRATION_HEADER_HEIGHT + MAX_PANEL_HEIGHT,
        :left_width,
    ] = image_panel
    pitch_panel = canvas[
        CALIBRATION_HEADER_HEIGHT:CALIBRATION_HEADER_HEIGHT + MAX_PANEL_HEIGHT,
        left_width:,
    ]
    rect = draw_pitch(pitch_panel, geometry)
    landmarks = named_landmarks(geometry)
    chosen = {item["landmark_id"] for item in correspondences}
    for landmark in landmarks:
        point = _pitch_to_panel(
            landmark["pitch_x_m"],
            landmark["pitch_y_m"],
            geometry,
            rect,
        )
        local_point = (point[0], point[1])
        color = (0, 255, 255) if landmark["landmark_id"] == selected_id else (
            (0, 200, 0) if landmark["landmark_id"] in chosen else (250, 250, 250)
        )
        cv2.circle(pitch_panel, local_point, 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            pitch_panel,
            landmark["landmark_id"],
            (local_point[0] + 7, local_point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    completed = len(correspondences)
    pending_text = (
        f"Now click image point for {selected_id}"
        if selected_id
        else "Click an unused named point on the pitch diagram first"
    )
    lines = [
        f"PITCH CALIBRATION | {segment_id} | reference frame {reference_frame}",
        "Left: reference image | right: canonical pitch diagram",
        "Click a named pitch marker, then its exact visible line intersection in the image.",
        pending_text,
        "Choose uniquely identifiable stationary intersections; avoid players and arbitrary line points.",
        f"Complete pairs: {completed} | minimum 4; 5+ needed for held-out validation.",
        "U: undo | R: reset | ENTER: estimate/save | ESC: cancel",
        "Default 105 x 68 m is assumed. Distinct camera segments need separate calibration.",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 18 + i * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49 if i else 0.55,
            (0, 255, 255) if i in (0, 2) else (245, 245, 245),
            1 if i else 2,
            cv2.LINE_AA,
        )
    for index, item in enumerate(correspondences):
        point = (
            int(round(item["image_x"] * left_width / frame.shape[1])),
            CALIBRATION_HEADER_HEIGHT
            + int(round(item["image_y"] * MAX_PANEL_HEIGHT / frame.shape[0])),
        )
        cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{index + 1}:{item['landmark_id']}",
            (point[0] + 7, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas, rect, landmarks


def select_calibration_points(frame, geometry, segment_id, reference_frame):
    correspondences = []
    pending_id = None
    landmarks = {item["landmark_id"]: item for item in named_landmarks(geometry)}
    state = {"pitch_rect": None, "pitch_panel_offset": 0, "landmarks": landmarks}

    def draw():
        canvas, rect, current_landmarks = _draw_calibration_ui(
            frame,
            geometry,
            correspondences,
            pending_id,
            segment_id,
            reference_frame,
        )
        state["pitch_rect"] = rect
        state["landmarks"] = {
            item["landmark_id"]: item for item in current_landmarks
        }
        return canvas

    def callback(event, x, y, flags, userdata):
        nonlocal pending_id
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        top = CALIBRATION_HEADER_HEIGHT
        bottom = top + MAX_PANEL_HEIGHT
        left_width = CALIBRATION_PANEL_WIDTH
        if y < top or y >= bottom:
            return
        if x >= left_width:
            if pending_id is not None:
                print("Complete the selected image point before choosing another pitch point.")
                return
            pitch_x = x - left_width
            pitch_y = y - top
            rect = state["pitch_rect"]
            clicked = None
            closest_distance = float("inf")
            for landmark in state["landmarks"].values():
                px, py = _pitch_to_panel(
                    landmark["pitch_x_m"], landmark["pitch_y_m"], geometry, rect
                )
                distance = math.hypot(px - pitch_x, py - pitch_y)
                if distance < closest_distance:
                    clicked = landmark
                    closest_distance = distance
            if clicked is None or closest_distance > 18:
                print("Click a named landmark marker on the pitch diagram.")
                return
            if any(
                row["landmark_id"] == clicked["landmark_id"]
                for row in correspondences
            ):
                print(f"{clicked['landmark_id']} is already paired; use U to undo.")
                return
            pending_id = clicked["landmark_id"]
            return
        if pending_id is None:
            print("Select the named canonical pitch landmark first.")
            return
        image_x = (x / left_width) * frame.shape[1]
        image_y = ((y - top) / MAX_PANEL_HEIGHT) * frame.shape[0]
        canonical = landmarks[pending_id]
        correspondences.append({
            "correspondence_id": len(correspondences) + 1,
            "landmark_id": pending_id,
            "landmark_name": canonical["landmark_name"],
            "reference_frame": reference_frame,
            "image_x": float(image_x),
            "image_y": float(image_y),
            "pitch_x_m": canonical["pitch_x_m"],
            "pitch_y_m": canonical["pitch_y_m"],
        })
        print(f"Paired {pending_id} ({canonical['landmark_name']}).")
        pending_id = None

    window = f"Pitch calibration | {segment_id} | frame {reference_frame}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, callback)
    print("\nNamed landmark calibration")
    print("Click a landmark on the pitch diagram, then the exact same visible intersection.")
    print("Do not select arbitrary points along a line or infer hidden landmarks.")
    print("U = undo, R = reset, ENTER = estimate/save, ESC = cancel.")
    try:
        while True:
            cv2.imshow(window, draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                if pending_id is not None:
                    print("Complete the pending image correspondence first.")
                elif len(correspondences) < MIN_CORRESPONDENCES:
                    print(
                        f"At least {MIN_CORRESPONDENCES} complete non-collinear "
                        "correspondences are required."
                    )
                else:
                    return correspondences
            elif key in (ord("u"), ord("U")):
                if pending_id is not None:
                    pending_id = None
                elif correspondences:
                    correspondences.pop()
                print("Undo completed.")
            elif key in (ord("r"), ord("R")):
                correspondences.clear()
                pending_id = None
                print("All correspondences cleared.")
            elif key == 27:
                raise RuntimeError("Pitch calibration cancelled.")
    finally:
        cv2.destroyAllWindows()


def _save_calibration(args, context, geometry, correspondences, estimation):
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT / f"calibration_{context['segment_id']}").resolve()
    )
    outputs, temporaries = _atomic_output_paths(output_dir, CALIBRATION_OUTPUTS)
    image_points = [
        (item["image_x"], item["image_y"]) for item in correspondences
    ]
    pitch_points = [
        (item["pitch_x_m"], item["pitch_y_m"]) for item in correspondences
    ]
    image_hull = cv2.convexHull(
        np.asarray(image_points, dtype=np.float32)
    ).reshape(-1, 2).tolist()
    pitch_hull = cv2.convexHull(
        np.asarray(pitch_points, dtype=np.float32)
    ).reshape(-1, 2).tolist()
    calibration = {
        "calibration_id": uuid4().hex,
        "status": estimation["status"],
        "valid_for_mapping": estimation["valid_for_mapping"],
        "source_video": str(context["video_path"]),
        "source_video_fps": context["fps"],
        "source_video_dimensions": [context["width"], context["height"]],
        "camera_transform_file": str(context["transforms_path"]),
        "camera_transform_sha256": _sha256(context["transforms_path"]),
        "camera_quality_file": str(context["quality_path"]),
        "camera_quality_sha256": _sha256(context["quality_path"]),
        "camera_segment_id": context["segment_id"],
        "reference_frame": context["reference_frame"],
        "tracking_run_id": context["tracking_run_id"],
        "pitch_coordinate_system": (
            "x=0 at goal line A, x=length at goal line B; y=0 at sideline A, "
            "y=width at sideline B; origin is the goal-line-A/sideline-A corner."
        ),
        "coordinate_convention": (
            "Image points are in original reference-frame pixels. The homography "
            "maps column vectors [x, y, 1] from this segment reference image to "
            "canonical pitch coordinates [x_m, y_m, 1]."
        ),
        "pitch_geometry": geometry,
        "candidate_landmarks": [
            item["landmark_name"] for item in named_landmarks(geometry)
        ],
        "pitch_dimensions_are_assumed": True,
        "dimension_note": (
            "Default 105 x 68 m is an assumed model, not a verified measurement "
            "of the stadium pitch."
        ),
        "homography_image_to_pitch": estimation["homography"],
        "correspondence_count": len(correspondences),
        "landmark_ids": [item["landmark_id"] for item in correspondences],
        "image_calibration_hull": image_hull,
        "pitch_calibration_hull": pitch_hull,
        "estimation_diagnostics": {
            key: value
            for key, value in estimation.items()
            if key not in {"homography"}
        },
        "mapping_scope": (
            "Only this camera segment and its reference-frame coordinate system. "
            "No cross-segment registration is implied."
        ),
    }
    try:
        with temporaries["pitch_calibration.json"].open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(calibration, file, indent=2)
            file.write("\n")
        _write_csv(
            temporaries["calibration_correspondences.csv"],
            CALIBRATION_FIELDS,
            correspondences,
        )
        diagnostic = draw_calibration_diagnostic(
            context["frame"],
            geometry,
            context["segment_id"],
            context["reference_frame"],
            "Calibration valid for mapping."
            if estimation["valid_for_mapping"]
            else "Calibration not valid for mapping; review diagnostics.",
            correspondences,
            estimation,
        )
        if not cv2.imwrite(
            str(temporaries["calibration_diagnostics.png"]), diagnostic
        ):
            raise RuntimeError("Could not write calibration diagnostic image.")
        for name in CALIBRATION_OUTPUTS:
            if outputs[name].exists():
                raise FileExistsError(f"Refusing to overwrite {outputs[name]}")
        for name in CALIBRATION_OUTPUTS:
            temporaries[name].replace(outputs[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        raise
    return calibration


def inspect_calibration(args):
    geometry = _geometry_from_args(args)
    context = _calibration_context(args)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT / f"inspection_{context['segment_id']}").resolve()
    )
    outputs, temporaries = _atomic_output_paths(output_dir, CALIBRATION_OUTPUTS)
    status = {
        "status": "manual_calibration_required",
        "valid_for_mapping": False,
        "source_video": str(context["video_path"]),
        "tracking_run_id": context["tracking_run_id"],
        "camera_segment_id": context["segment_id"],
        "reference_frame": context["reference_frame"],
        "pitch_geometry": geometry,
        "pitch_dimensions_are_assumed": True,
        "correspondence_count": 0,
        "candidate_landmarks": [
            item["landmark_name"] for item in named_landmarks(geometry)
        ],
        "homography_image_to_pitch": None,
        "diagnostic": (
            "No manually selected correspondences were available. This inspection "
            "does not estimate or invent a pitch mapping."
        ),
    }
    try:
        with temporaries["pitch_calibration.json"].open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(status, file, indent=2)
            file.write("\n")
        _write_csv(
            temporaries["calibration_correspondences.csv"],
            CALIBRATION_FIELDS,
            [],
        )
        diagnostic = draw_calibration_diagnostic(
            context["frame"],
            geometry,
            context["segment_id"],
            context["reference_frame"],
            "MANUAL CALIBRATION REQUIRED - no homography estimated",
        )
        if not cv2.imwrite(
            str(temporaries["calibration_diagnostics.png"]), diagnostic
        ):
            raise RuntimeError("Could not write calibration diagnostic image.")
        for name in CALIBRATION_OUTPUTS:
            temporaries[name].replace(outputs[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        raise
    return status


def calibrate_segment(args):
    geometry = _geometry_from_args(args)
    context = _calibration_context(args)
    correspondences = select_calibration_points(
        context["frame"], geometry, context["segment_id"], context["reference_frame"]
    )
    for item in correspondences:
        item["run_id"] = context["tracking_run_id"]
        item["camera_segment_id"] = context["segment_id"]
    landmarks = {item["landmark_id"]: item for item in named_landmarks(geometry)}
    point_by_id = {item["landmark_id"]: item for item in correspondences}
    ordered = [point_by_id[key] for key in point_by_id]
    estimation = estimate_homography(
        [item["landmark_id"] for item in ordered],
        [(item["image_x"], item["image_y"]) for item in ordered],
        [(item["pitch_x_m"], item["pitch_y_m"]) for item in ordered],
        context["width"],
        context["height"],
        geometry,
        args.ransac_threshold_m,
    )
    calibration = _save_calibration(args, context, geometry, ordered, estimation)
    print(f"Calibration status: {calibration['status']}")
    print(f"Valid for mapping: {calibration['valid_for_mapping']}")
    if estimation["errors"]:
        print("Calibration warnings: " + ", ".join(estimation["errors"]))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT / f"calibration_{context['segment_id']}").resolve()
    )
    for name in CALIBRATION_OUTPUTS:
        print(f"Saved: {output_dir / name}")
    return calibration


def load_calibrations(calibration_dir, video_path, geometry):
    calibration_files = sorted(calibration_dir.rglob("pitch_calibration.json"))
    if not calibration_files:
        raise FileNotFoundError(
            f"No pitch_calibration.json found under {calibration_dir}."
        )
    calibrations = {}
    for path in calibration_files:
        item = _read_json(path, "Pitch calibration")
        if not item.get("valid_for_mapping"):
            continue
        if Path(item.get("source_video", "")).expanduser().resolve() != video_path:
            raise ValueError(f"Calibration {path} refers to a different source video.")
        if item.get("pitch_geometry") != geometry:
            raise ValueError(f"Calibration {path} uses different pitch dimensions.")
        segment = item.get("camera_segment_id")
        reference = item.get("reference_frame")
        matrix = np.asarray(item.get("homography_image_to_pitch"), dtype=np.float64)
        error, _ = _homography_condition(matrix)
        if error:
            raise ValueError(f"Calibration {path} contains invalid homography: {error}")
        if segment in calibrations:
            raise ValueError(f"Multiple valid calibrations found for {segment}.")
        item["_matrix"] = matrix
        item["_path"] = str(path)
        item["_hull"] = np.asarray(
            item.get("image_calibration_hull", []), dtype=np.float32
        )
        if item["_hull"].ndim != 2 or item["_hull"].shape[1:] != (2,):
            raise ValueError(f"Calibration {path} has invalid image-region hull.")
        if len(item["_hull"]) < 3:
            raise ValueError(f"Calibration {path} has too few hull points.")
        calibrations[segment] = item
    return calibrations


def _inside_calibration_hull(calibration, x, y):
    point = (float(x), float(y))
    contour = calibration["_hull"].reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def map_compensated_rows(rows, assignments, calibrations, geometry, boundary_tolerance=1.0):
    if not math.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and non-negative.")
    mapped = []
    invalid_reasons = Counter()
    for source in rows:
        key = (source["run_id"], source["track_id"])
        assignment = assignments.get(key, {})
        role = assignment.get("predicted_role", "unknown") or "unknown"
        predicted_team = assignment.get("predicted_team", "unknown") or "unknown"
        row = {
            "run_id": source["run_id"],
            "frame": source["frame"],
            "timestamp": source["timestamp"],
            "track_id": source["track_id"],
            "class_id": source["class_id"],
            "confidence": source["confidence"],
            "x1": source["x1"],
            "y1": source["y1"],
            "x2": source["x2"],
            "y2": source["y2"],
            "center_x": source.get("center_x", ""),
            "center_y": source.get("center_y", ""),
            "foot_x": source["foot_x"],
            "foot_y": source["foot_y"],
            "compensated_x": source["compensated_x"],
            "compensated_y": source["compensated_y"],
            "camera_segment_id": source["camera_segment_id"],
            "reference_frame": source["reference_frame"],
            "compensation_valid": source["compensation_valid"],
            "compensation_invalid_reason": source["invalid_reason"],
            "predicted_team": predicted_team,
            "predicted_role": role,
            "classification_confidence": assignment.get(
                "classification_confidence", ""
            ),
            "classification_samples": assignment.get("number_of_samples", 0),
            "classification_method": assignment.get(
                "classification_method", "missing_assignment"
            ),
            "pitch_x_m": "",
            "pitch_y_m": "",
            "pitch_mapping_valid": "False",
            "pitch_mapping_invalid_reason": "",
        }
        reason = ""
        if source["class_id"] != 0:
            reason = "not_person_detection"
        elif source["compensation_valid"].strip().lower() not in {
            "true", "1", "yes"
        }:
            reason = source["invalid_reason"] or "camera_compensation_invalid"
        elif not source["track_id"]:
            reason = "missing_track_id"
        else:
            calibration = calibrations.get(source["camera_segment_id"])
            if calibration is None:
                reason = "segment_calibration_missing"
            elif calibration["reference_frame"] != source["reference_frame"]:
                reason = "calibration_reference_frame_mismatch"
            elif not source["compensated_x"] or not source["compensated_y"]:
                reason = "missing_compensated_position"
            else:
                try:
                    x = float(source["compensated_x"])
                    y = float(source["compensated_y"])
                except (TypeError, ValueError):
                    reason = "invalid_compensated_position"
                else:
                    if not math.isfinite(x) or not math.isfinite(y):
                        reason = "nonfinite_compensated_position"
                    elif not _inside_calibration_hull(calibration, x, y):
                        reason = "outside_calibration_region"
                    else:
                        point, projection_error = project_homography(
                            calibration["_matrix"], x, y
                        )
                        if projection_error:
                            reason = projection_error
                        elif (
                            point[0] < -boundary_tolerance
                            or point[0] > geometry["length_m"] + boundary_tolerance
                            or point[1] < -boundary_tolerance
                            or point[1] > geometry["width_m"] + boundary_tolerance
                        ):
                            reason = "outside_pitch_bounds"
                        else:
                            row["pitch_x_m"] = point[0]
                            row["pitch_y_m"] = point[1]
                            row["pitch_mapping_valid"] = "True"
        row["pitch_mapping_invalid_reason"] = reason
        if reason:
            invalid_reasons[reason] += 1
        mapped.append(row)
    return mapped, invalid_reasons


def _draw_mapped_frame(frame, frame_rows, calibration, geometry, pitch_rect):
    left = cv2.resize(frame, (960, 540))
    right = np.zeros((540, 960, 3), dtype=np.uint8)
    pitch_rect = draw_pitch(right, geometry, calibration=calibration)
    cv2.putText(
        right,
        f"TOP-DOWN | {calibration['camera_segment_id']} ref={calibration['reference_frame']}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for row in frame_rows:
        if row["class_id"] != 0:
            continue
        role = row["predicted_role"]
        color = TEAM_COLORS.get(role, TEAM_COLORS["unknown"])
        x1, y1, x2, y2 = [
            int(round(float(row[name]))) for name in ("x1", "y1", "x2", "y2")
        ]
        sx, sy = left.shape[1] / frame.shape[1], left.shape[0] / frame.shape[0]
        cv2.rectangle(
            left,
            (int(x1 * sx), int(y1 * sy)),
            (int(x2 * sx), int(y2 * sy)),
            color,
            2,
        )
        cv2.putText(
            left,
            f"ID {row['track_id']} | {role}",
            (int(x1 * sx), max(16, int(y1 * sy) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
        if row["pitch_mapping_valid"] != "True":
            continue
        point = _pitch_to_panel(
            row["pitch_x_m"], row["pitch_y_m"], geometry, pitch_rect
        )
        cv2.circle(right, point, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            right,
            str(row["track_id"]),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        left,
        "IMAGE PIXELS | tracking boxes (not pitch coordinates)",
        (12, 526),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        "105 x 68 m assumed unless verified",
        (12, 526),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.hconcat([left, right])


def _render_mapping_video(
    video_path, start_frame, end_frame, fps, width, height, frame_segments, rows_by_frame,
    calibrations, geometry, output_path, preview_path,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
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
        capture.release()
        raise ValueError("Source video metadata does not match the tracking run.")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
        capture.release()
        raise RuntimeError(f"Cannot seek to source frame {start_frame}.")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1920, 540),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not initialize pitch mapping video: {output_path}")
    preview = None
    previous_segment = None
    try:
        for frame_number in tqdm(
            range(start_frame, end_frame),
            desc="Rendering pitch mapping",
            unit="frame",
        ):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read source frame {frame_number}.")
            current_rows = rows_by_frame.get(frame_number, [])
            segment = frame_segments[frame_number]
            if segment != previous_segment:
                previous_segment = segment
            calibration = calibrations.get(segment) if segment else None
            if calibration is None:
                left = cv2.resize(frame, (960, 540))
                right = np.zeros((540, 960, 3), dtype=np.uint8)
                draw_pitch(right, geometry)
                cv2.putText(
                    right,
                    f"No valid calibration for {segment or 'this frame'}",
                    (18, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    right,
                    "Pitch positions intentionally unavailable",
                    (18, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                comparison = cv2.hconcat([left, right])
            else:
                comparison = _draw_mapped_frame(
                    frame, current_rows, calibration, geometry, None
                )
            writer.write(comparison)
            if frame_number == end_frame - 1:
                preview = comparison
    finally:
        writer.release()
        capture.release()
    if preview is None:
        raise RuntimeError("No frames were rendered for the pitch mapping preview.")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Could not write pitch mapping preview: {preview_path}")


def map_tracking_run(args):
    geometry = _geometry_from_args(args)
    compensated_path = args.input_compensated.expanduser().resolve()
    team_path = args.team_assignments.expanduser().resolve()
    for path, name in (
        (compensated_path, "Compensated tracking CSV"),
        (team_path, "Team assignments CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    (
        comp_summary, tracking_summary, team_summary, run_id, video_path
    ) = load_tracking_context(compensated_path, team_path, args.video)
    rows = read_compensated_rows(compensated_path, run_id)
    assignments = read_team_assignments(team_path, run_id)
    calibrations = load_calibrations(
        args.calibration_dir.expanduser().resolve(), video_path, geometry
    )

    transform_path = Path(comp_summary["camera_transform_file"]).expanduser().resolve()
    quality_path = Path(comp_summary["camera_quality_file"]).expanduser().resolve()
    if not transform_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError("Camera transform or quality file from summary is missing.")
    for calibration in calibrations.values():
        if calibration.get("camera_transform_sha256") != _sha256(transform_path):
            raise ValueError("Calibration was created from different camera transforms.")
        if calibration.get("camera_quality_sha256") != _sha256(quality_path):
            raise ValueError("Calibration was created from different camera quality data.")

    mapped_rows, invalid_reasons = map_compensated_rows(
        rows,
        assignments,
        calibrations,
        geometry,
        args.calibration_boundary_tolerance_m,
    )
    try:
        start_frame, end_frame = map(
            int, comp_summary["tracking_frame_range"]
        )
        fps = float(comp_summary["source_video_fps"])
        width = int(comp_summary["source_frame_width"])
        height = int(comp_summary["source_frame_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Compensation summary lacks source video frame metadata.") from exc
    if end_frame <= start_frame or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Compensation summary contains invalid frame metadata.")
    for calibration in calibrations.values():
        if calibration.get("source_video_dimensions") != [width, height]:
            raise ValueError("Calibration image dimensions differ from mapped video.")
        if abs(float(calibration.get("source_video_fps", 0)) - fps) > 1e-3:
            raise ValueError("Calibration frame rate differs from mapped video.")
    for row_number, row in enumerate(rows, start=2):
        if not start_frame <= row["frame"] < end_frame:
            raise ValueError(
                f"Compensated row {row_number} is outside the summarized frame range."
            )
        if abs(row["timestamp"] - row["frame"] / fps) > 1e-3:
            raise ValueError(
                f"Compensated row {row_number} timestamp does not match frame/FPS."
            )
    transforms, quality = compensation.load_camera_data(transform_path, quality_path)
    camera_frames = compensation.build_frame_transforms(transforms, quality)
    if len(camera_frames) < end_frame:
        raise ValueError("Camera transforms do not cover the compensated frame range.")
    segment_references = {}
    for frame_number, frame_transform in enumerate(camera_frames):
        segment_references.setdefault(
            frame_transform.segment_id, frame_transform.reference_frame
        )
    frame_segments = [item.segment_id for item in camera_frames]
    for segment, calibration in calibrations.items():
        if segment not in segment_references:
            raise ValueError(f"Calibration references unknown camera segment {segment}.")
        if calibration["reference_frame"] != segment_references[segment]:
            raise ValueError(
                f"Calibration reference frame disagrees with camera segment {segment}."
            )
    for row in mapped_rows:
        if row["camera_segment_id"] in calibrations:
            expected_reference = calibrations[row["camera_segment_id"]]["reference_frame"]
            if row["reference_frame"] != expected_reference:
                row["pitch_mapping_valid"] = "False"
                row["pitch_x_m"] = ""
                row["pitch_y_m"] = ""
                row["pitch_mapping_invalid_reason"] = (
                    "calibration_reference_frame_mismatch"
                )
                invalid_reasons["calibration_reference_frame_mismatch"] += 1

    observation_frames = defaultdict(list)
    for row in mapped_rows:
        observation_frames[int(row["frame"])].append(row)
    persons = [row for row in mapped_rows if int(row["class_id"]) == 0]
    valid = [row for row in persons if row["pitch_mapping_valid"] == "True"]
    encountered_segments = sorted({
        row["camera_segment_id"] for row in mapped_rows
    })
    summary = {
        "mapping_run_id": uuid4().hex,
        "tracking_run_id": run_id,
        "compensation_run_id": comp_summary.get("compensation_run_id"),
        "team_classification_run_id": team_summary.get("classification_run_id"),
        "input_video": str(video_path),
        "compensated_tracking_csv": str(compensated_path),
        "team_assignments_csv": str(team_path),
        "track_set": "confirmed",
        "pitch_geometry": geometry,
        "pitch_dimensions_are_assumed": True,
        "pitch_coordinate_system": (
            "x=0 at goal line A, x=length at goal line B; y=0 at sideline A, "
            "y=width at sideline B."
        ),
        "dimension_note": (
            "The default 105 x 68 m dimensions are assumptions, not verified "
            "stadium measurements."
        ),
        "camera_segment_ids_encountered": encountered_segments,
        "calibrated_camera_segments": sorted(calibrations),
        "camera_segments_without_calibration": sorted(
            set(encountered_segments) - set(calibrations)
        ),
        "player_observations": len(persons),
        "successfully_mapped_observations": len(valid),
        "unmapped_observations": len(persons) - len(valid),
        "unmapped_reasons": dict(sorted(invalid_reasons.items())),
        "mapped_track_ids": len({
            (row["run_id"], row["track_id"]) for row in valid
        }),
        "coordinate_scope": (
            "Each camera segment uses its own reference-frame calibration. "
            "Coordinates from distinct segments are not comparable unless "
            "independently registered to a shared pitch coordinate system."
        ),
        "classifications_are_provisional": True,
        "outputs": list(MAPPING_OUTPUTS),
    }
    output_dir = args.output_dir.expanduser().resolve()
    outputs, temporaries = _atomic_output_paths(output_dir, MAPPING_OUTPUTS)
    try:
        _write_csv(
            temporaries["mapped_player_positions.csv"],
            MAPPING_FIELDS,
            mapped_rows,
        )
        with temporaries["mapping_summary.json"].open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
        _render_mapping_video(
            video_path,
            start_frame,
            end_frame,
            fps,
            width,
            height,
            frame_segments,
            observation_frames,
            calibrations,
            geometry,
            temporaries["annotated_pitch_mapping.mp4"],
            temporaries["pitch_mapping_preview.png"],
        )
        for name in MAPPING_OUTPUTS:
            if outputs[name].exists():
                raise FileExistsError(f"Refusing to overwrite {outputs[name]}")
        for name in MAPPING_OUTPUTS:
            temporaries[name].replace(outputs[name])
    except Exception:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
        raise
    return summary


def validate_args(args):
    if args.command in {"inspect", "calibrate"}:
        if not args.segment_id:
            raise ValueError("--segment-id cannot be empty.")
        _geometry_from_args(args)
        if args.command == "calibrate" and (
            not math.isfinite(args.ransac_threshold_m)
            or args.ransac_threshold_m <= 0
        ):
            raise ValueError("--ransac-threshold-m must be finite and positive.")
    elif args.command == "map":
        _geometry_from_args(args)
        if (
            not math.isfinite(args.calibration_boundary_tolerance_m)
            or args.calibration_boundary_tolerance_m < 0
        ):
            raise ValueError(
                "--calibration-boundary-tolerance-m must be finite and non-negative."
            )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args.command == "inspect":
            result = inspect_calibration(args)
            print(
                "No homography estimated. Inspect the reference frame and run "
                "calibrate after selecting identifiable landmarks."
            )
        elif args.command == "calibrate":
            result = calibrate_segment(args)
        else:
            result = map_tracking_run(args)
            print(
                f"Mapped player observations: {result['successfully_mapped_observations']} "
                f"/ {result['player_observations']}"
            )
        if args.command != "map":
            print(f"Status: {result['status']}")
            print(f"Valid for mapping: {result['valid_for_mapping']}")
    except (
        cv2.error,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        csv.Error,
        np.linalg.LinAlgError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
