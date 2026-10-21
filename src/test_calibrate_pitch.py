import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_pitch as pitch


def synthetic_correspondences(matrix=None, count=6):
    matrix = np.asarray(
        matrix
        if matrix is not None
        else [
            [105.0 / 1280.0, 0.0, 0.0],
            [0.0, 68.0 / 720.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image = np.asarray(
        [
            [80, 70],
            [1190, 90],
            [100, 650],
            [1180, 660],
            [600, 80],
            [650, 620],
            [360, 350],
        ][:count],
        dtype=np.float64,
    )
    target = []
    for x, y in image:
        point = matrix @ np.array([x, y, 1.0])
        target.append(point[:2] / point[2])
    return image, np.asarray(target), [f"L{i:02d}" for i in range(1, count + 1)]


def compensated_row(
    frame=0,
    track_id="7",
    run_id="run-a",
    segment="segment_000",
    reference=0,
    point=(640.0, 360.0),
    valid="True",
    class_id=0,
):
    x, y = point
    return {
        "run_id": run_id,
        "frame": frame,
        "timestamp": frame / 25.0,
        "track_id": track_id,
        "class_id": class_id,
        "confidence": "0.9",
        "x1": "620",
        "y1": "300",
        "x2": "660",
        "y2": "360",
        "center_x": "640",
        "center_y": "330",
        "foot_x": str(x),
        "foot_y": str(y),
        "compensated_x": str(x) if valid == "True" else "",
        "compensated_y": str(y) if valid == "True" else "",
        "camera_segment_id": segment,
        "reference_frame": reference,
        "compensation_valid": valid,
        "invalid_reason": "" if valid == "True" else "failed_transition",
    }


def assignment(run_id="run-a", track_id="7", role="team_a_outfield"):
    return {
        "run_id": run_id,
        "track_id": track_id,
        "predicted_team": "team_a" if role == "team_a_outfield" else "unknown",
        "predicted_role": role,
        "classification_confidence": "0.8",
        "number_of_samples": "12",
        "classification_method": "test",
    }


def calibration_for(segment="segment_000", reference=0, matrix=None, hull=None):
    if matrix is None:
        matrix = np.array([
            [105.0 / 1280.0, 0.0, 0.0],
            [0.0, 68.0 / 720.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    if hull is None:
        hull = [[0, 0], [1279, 0], [1279, 719], [0, 719]]
    return {
        "camera_segment_id": segment,
        "reference_frame": reference,
        "_matrix": np.asarray(matrix, dtype=np.float64),
        "_hull": np.asarray(hull, dtype=np.float32),
    }


class CalibratePitchTests(unittest.TestCase):
    def setUp(self):
        self.geometry = pitch.pitch_geometry()

    def test_default_pitch_geometry_and_named_spots(self):
        landmarks = {row["landmark_name"]: row for row in pitch.named_landmarks(self.geometry)}
        self.assertEqual(self.geometry["length_m"], 105.0)
        self.assertEqual(self.geometry["width_m"], 68.0)
        self.assertEqual(
            (landmarks["center_spot"]["pitch_x_m"], landmarks["center_spot"]["pitch_y_m"]),
            (52.5, 34.0),
        )
        self.assertEqual(landmarks["penalty_spot_left"]["pitch_x_m"], 11.0)
        self.assertEqual(landmarks["penalty_spot_right"]["pitch_x_m"], 94.0)
        self.assertEqual(len({x["landmark_id"] for x in landmarks.values()}), len(landmarks))

    def test_custom_geometry_validation(self):
        geometry = pitch.pitch_geometry(length=100, width=64)
        self.assertEqual(geometry["length_m"], 100)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            pitch.pitch_geometry(length=20, width=10)

    def test_identity_homography_projection(self):
        self.assertEqual(
            pitch.project_homography(np.eye(3), 321, 123),
            ((321.0, 123.0), ""),
        )

    def test_translation_homography_projection(self):
        matrix = np.array([[1, 0, 5], [0, 1, -8], [0, 0, 1]], dtype=float)
        projected, error = pitch.project_homography(matrix, 3, 10)
        self.assertEqual(error, "")
        self.assertEqual(projected, (8.0, 2.0))

    def test_rotation_homography_projection(self):
        matrix = np.array([[0, -1, 720], [1, 0, 0], [0, 0, 1]], dtype=float)
        projected, error = pitch.project_homography(matrix, 20, 30)
        self.assertEqual(error, "")
        self.assertEqual(projected, (690.0, 20.0))

    def test_perspective_homography_projection(self):
        matrix = np.array(
            [[1, 0.1, 2], [0.05, 1, 3], [0.001, 0.002, 1]], dtype=float
        )
        projected, error = pitch.project_homography(matrix, 10, 20)
        self.assertEqual(error, "")
        self.assertTrue(np.allclose(projected, (14 / 1.05, 23.5 / 1.05)))

    def test_synthetic_image_to_pitch_homography_and_held_out_validation(self):
        image, target, ids = synthetic_correspondences()
        result = pitch.estimate_homography(
            ids, image, target, 1280, 720, self.geometry
        )
        self.assertTrue(result["valid_for_mapping"], result["errors"])
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["held_out_landmark_id"], ids[-1])
        self.assertLess(result["held_out_error_m"], 1e-6)
        matrix = np.asarray(result["homography"])
        for source, expected in zip(image, target):
            actual, error = pitch.project_homography(matrix, *source)
            self.assertEqual(error, "")
            self.assertTrue(np.allclose(actual, expected, atol=1e-5))

    def test_homography_composition_maps_compensation_before_pitch(self):
        camera_compensation = np.array(
            [[1, 0, -100], [0, 1, 30], [0, 0, 1]], dtype=float
        )
        image_to_pitch = np.array(
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=float
        )
        image_point = np.array([500, 300, 1], dtype=float)
        compensated = camera_compensation @ image_point
        expected = image_to_pitch @ compensated
        correct = image_to_pitch @ camera_compensation @ image_point
        wrong = camera_compensation @ image_to_pitch @ image_point
        self.assertTrue(np.allclose(expected / expected[2], [40, 33, 1]))
        self.assertFalse(np.allclose(correct, wrong))

    def test_four_correspondences_are_not_accepted_without_held_out_check(self):
        image, target, ids = synthetic_correspondences(count=4)
        result = pitch.estimate_homography(
            ids, image, target, 1280, 720, self.geometry
        )
        self.assertFalse(result["valid_for_mapping"])
        self.assertEqual(result["status"], "insufficient_independent_validation")
        self.assertIn(
            "held_out_validation_requires_at_least_five_points", result["errors"]
        )

    def test_collinear_and_degenerate_correspondences_are_rejected(self):
        image = np.array([[100, 100], [200, 200], [300, 300], [400, 400]], float)
        target = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], float)
        with self.assertRaisesRegex(ValueError, "collinear"):
            pitch.validate_correspondences(
                ["a", "b", "c", "d"], image, target, 1280, 720, self.geometry
            )

    def test_duplicate_and_mismatched_correspondences_are_rejected(self):
        image, target, ids = synthetic_correspondences(count=5)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pitch.validate_correspondences(
                ids[:4], image, target, 1280, 720, self.geometry
            )
        with self.assertRaisesRegex(ValueError, "only once"):
            pitch.validate_correspondences(
                ["x", "x", "y", "z", "w"],
                image,
                target,
                1280,
                720,
                self.geometry,
            )

    def test_out_of_bounds_image_landmark_rejected(self):
        image, target, ids = synthetic_correspondences(count=5)
        image[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "inside the frame"):
            pitch.validate_correspondences(
                ids, image, target, 1280, 720, self.geometry
            )

    def test_nonfinite_singular_and_ill_conditioned_homographies_rejected(self):
        self.assertEqual(
            pitch._homography_condition(np.full((3, 3), np.nan))[0],
            "nonfinite_homography",
        )
        self.assertEqual(
            pitch._homography_condition(np.diag([1, 1, 0]))[0],
            "singular_homography",
        )
        self.assertEqual(
            pitch._homography_condition(np.diag([1, 0.5, 1e-13]))[0],
            "ill_conditioned_homography",
        )

    def test_near_zero_homogeneous_denominator_rejected(self):
        matrix = np.diag([1.0, 1.0, 1e-14])
        point, error = pitch.project_homography(matrix, 10, 20)
        self.assertIsNone(point)
        self.assertEqual(error, "near_zero_homogeneous_denominator")

    def test_reversed_viewpoint_orientation_is_reported_for_manual_confirmation(self):
        reflected = np.array(
            [[-105 / 1280, 0, 105], [0, 68 / 720, 0], [0, 0, 1]],
            dtype=float,
        )
        image, target, ids = synthetic_correspondences(reflected)
        result = pitch.estimate_homography(
            ids, image, target, 1280, 720, self.geometry
        )
        self.assertTrue(result["valid_for_mapping"], result["errors"])
        self.assertEqual(result["geographic_orientation"], "negative_image_to_pitch")
        self.assertIn("verify named goal and sideline", result["orientation_warning"])

    def test_pixel_points_outside_source_frame_rejected(self):
        _, target, ids = synthetic_correspondences()
        image = np.array([[2000, 100], [1190, 90], [100, 650], [1180, 660], [600, 80], [650, 620]])
        with self.assertRaisesRegex(ValueError, "inside the frame"):
            pitch.validate_correspondences(
                ids, image, target, 1280, 720, self.geometry
            )

    def test_run_scoped_team_join_does_not_join_by_track_id_only(self):
        row = compensated_row(run_id="run-b")
        mapped, _ = pitch.map_compensated_rows(
            [row],
            {("run-a", "7"): assignment(run_id="run-a")},
            {"segment_000": calibration_for()},
            self.geometry,
        )
        self.assertEqual(mapped[0]["predicted_role"], "unknown")
        self.assertEqual(mapped[0]["classification_method"], "missing_assignment")

    def test_team_csv_rejects_mixed_runs_and_duplicate_track_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "team_assignments.csv"
            rows = [
                assignment("run-a", "1"),
                assignment("run-b", "2"),
            ]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "run ID mismatch"):
                pitch.read_team_assignments(path, "run-a")
            rows = [assignment("run-a", "1"), assignment("run-a", "1")]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                pitch.read_team_assignments(path, "run-a")

    def test_valid_mapping_preserves_run_frames_and_classification(self):
        row = compensated_row(frame=17, point=(640, 360))
        mapped, reasons = pitch.map_compensated_rows(
            [row],
            {("run-a", "7"): assignment()},
            {"segment_000": calibration_for()},
            self.geometry,
        )
        result = mapped[0]
        self.assertEqual(reasons, {})
        self.assertEqual(result["run_id"], "run-a")
        self.assertEqual(result["frame"], 17)
        self.assertEqual(result["timestamp"], 17 / 25)
        self.assertEqual(result["track_id"], "7")
        self.assertEqual(result["predicted_team"], "team_a")
        self.assertEqual(result["predicted_role"], "team_a_outfield")
        self.assertEqual(result["camera_segment_id"], "segment_000")
        self.assertEqual(result["reference_frame"], 0)
        self.assertEqual(result["pitch_mapping_valid"], "True")
        self.assertAlmostEqual(result["pitch_x_m"], 52.5)
        self.assertAlmostEqual(result["pitch_y_m"], 34.0)

    def test_camera_segment_boundaries_require_independent_calibration(self):
        rows = [
            compensated_row(1, segment="segment_000", reference=0),
            compensated_row(2, segment="segment_001", reference=2),
        ]
        mapped, reasons = pitch.map_compensated_rows(
            rows,
            {("run-a", "7"): assignment()},
            {"segment_000": calibration_for("segment_000", 0)},
            self.geometry,
        )
        self.assertEqual(mapped[0]["pitch_mapping_valid"], "True")
        self.assertEqual(mapped[1]["pitch_mapping_valid"], "False")
        self.assertEqual(
            mapped[1]["pitch_mapping_invalid_reason"],
            "segment_calibration_missing",
        )
        self.assertEqual(reasons["segment_calibration_missing"], 1)

    def test_invalid_compensation_and_reference_mismatch_stay_unmapped(self):
        rows = [
            compensated_row(1, valid="False"),
            compensated_row(2, reference=3),
        ]
        mapped, _ = pitch.map_compensated_rows(
            rows,
            {("run-a", "7"): assignment()},
            {"segment_000": calibration_for(reference=0)},
            self.geometry,
        )
        self.assertEqual(
            mapped[0]["pitch_mapping_invalid_reason"], "failed_transition"
        )
        self.assertEqual(
            mapped[1]["pitch_mapping_invalid_reason"],
            "calibration_reference_frame_mismatch",
        )

    def test_out_of_calibration_region_is_not_extrapolated(self):
        hull = [[500, 250], [800, 250], [800, 500], [500, 500]]
        row = compensated_row(point=(100, 100))
        mapped, reasons = pitch.map_compensated_rows(
            [row],
            {("run-a", "7"): assignment()},
            {"segment_000": calibration_for(hull=hull)},
            self.geometry,
        )
        self.assertEqual(mapped[0]["pitch_mapping_valid"], "False")
        self.assertEqual(
            mapped[0]["pitch_mapping_invalid_reason"], "outside_calibration_region"
        )
        self.assertEqual(reasons["outside_calibration_region"], 1)

    def test_out_of_pitch_projection_is_not_reported_as_valid(self):
        matrix = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]], dtype=float)
        row = compensated_row(point=(640, 360))
        mapped, _ = pitch.map_compensated_rows(
            [row],
            {("run-a", "7"): assignment()},
            {"segment_000": calibration_for(matrix=matrix)},
            self.geometry,
        )
        self.assertEqual(mapped[0]["pitch_mapping_valid"], "False")
        self.assertEqual(
            mapped[0]["pitch_mapping_invalid_reason"], "outside_pitch_bounds"
        )

    def test_nonperson_and_empty_inputs_are_preserved_safely(self):
        ball = compensated_row(class_id=32, track_id="", point=(10, 10))
        rows, reasons = pitch.map_compensated_rows(
            [ball], {}, {}, self.geometry
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pitch_mapping_invalid_reason"], "not_person_detection")
        empty, empty_reasons = pitch.map_compensated_rows(
            [], {}, {}, self.geometry
        )
        self.assertEqual(empty, [])
        self.assertEqual(empty_reasons, {})

    def test_mapping_output_schema_and_refuse_overwrite(self):
        self.assertEqual(
            pitch.MAPPING_FIELDS,
            [
                "run_id", "frame", "timestamp", "track_id", "class_id",
                "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
                "foot_x", "foot_y", "compensated_x", "compensated_y",
                "camera_segment_id", "reference_frame", "compensation_valid",
                "compensation_invalid_reason", "predicted_team", "predicted_role",
                "classification_confidence", "classification_samples",
                "classification_method", "pitch_x_m", "pitch_y_m",
                "pitch_mapping_valid", "pitch_mapping_invalid_reason",
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapped.csv"
            pitch._write_csv(path, pitch.MAPPING_FIELDS, [{
                "run_id": "run-a",
                "frame": 9,
                "timestamp": 0.36,
                "track_id": "7",
                "camera_segment_id": "segment_000",
                "pitch_mapping_valid": "False",
                "pitch_mapping_invalid_reason": "segment_calibration_missing",
            }])
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, pitch.MAPPING_FIELDS)
                row = next(reader)
                self.assertEqual(row["frame"], "9")
                self.assertEqual(row["pitch_mapping_invalid_reason"], "segment_calibration_missing")
        with tempfile.TemporaryDirectory() as temp:
            output, temporary = pitch._atomic_output_paths(
                Path(temp), pitch.MAPPING_OUTPUTS
            )
            for path in temporary.values():
                path.unlink(missing_ok=True)
            output["mapped_player_positions.csv"].write_text(
                "historical", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                pitch._atomic_output_paths(Path(temp), pitch.MAPPING_OUTPUTS)
            self.assertEqual(
                output["mapped_player_positions.csv"].read_text(encoding="utf-8"),
                "historical",
            )

    def test_calibration_file_loader_ignores_pending_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pending = root / "inspect"
            pending.mkdir()
            (pending / "pitch_calibration.json").write_text(json.dumps({
                "status": "manual_calibration_required",
                "valid_for_mapping": False,
            }), encoding="utf-8")
            self.assertEqual(
                pitch.load_calibrations(root, Path("video.mp4").resolve(), self.geometry),
                {},
            )


if __name__ == "__main__":
    unittest.main()
