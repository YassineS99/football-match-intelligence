import tempfile
import unittest
from pathlib import Path

import numpy as np

import calibrate_pitch as pitch
import compensate_player_tracks as compensation
import extend_pitch_calibration as extension


class MultiFrameCalibrationTests(unittest.TestCase):
    def test_noncommuting_camera_steps_compose_into_reference_order(self):
        step_one = np.array([[1, 0, 12], [0, 1, -4], [0, 0, 1]], dtype=float)
        step_two = np.array([[2, 0, 3], [0, 0.5, 7], [0, 0, 1]], dtype=float)
        transforms = np.asarray([np.eye(3), step_one, step_two])
        quality = [
            {"frame": i, "method": "orb", "valid": True}
            for i in range(len(transforms))
        ]
        frame_transforms = compensation.build_frame_transforms(transforms, quality)
        matrix, _ = extension.compose_frame_chain(
            frame_transforms, 2, "segment_000", 0
        )
        point = np.array([25.0, 30.0, 1.0])
        expected = step_one @ step_two @ point
        wrong = step_two @ step_one @ point
        self.assertTrue(np.allclose(matrix @ point, expected))
        self.assertFalse(np.allclose(expected, wrong))

    def test_landmark_raw_to_reference_projection(self):
        matrix = np.array([[2, 0, 3], [0, 0.5, 7], [0, 0, 1]], dtype=float)
        self.assertEqual(
            extension.transform_landmark_point(matrix, 25, 30),
            (53.0, 22.0),
        )

    def test_camera_failure_cannot_be_crossed_or_joined(self):
        transforms = np.asarray([np.eye(3), np.eye(3), np.eye(3)])
        quality = [
            {"frame": 0, "method": "identity", "valid": True},
            {"frame": 1, "method": "orb", "valid": True},
            {"frame": 2, "method": "failed", "valid": False},
        ]
        frame_transforms = compensation.build_frame_transforms(transforms, quality)
        with self.assertRaisesRegex(ValueError, "not requested segment"):
            extension.compose_frame_chain(frame_transforms, 2, "segment_000", 0)
        self.assertEqual(frame_transforms[2].segment_id, "segment_001")
        self.assertEqual(frame_transforms[2].reference_frame, 2)

    def test_nonfinite_and_singular_camera_transforms_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite 3x3"):
            extension.transform_landmark_point(
                np.array([[1, 0, 0], [0, np.nan, 0], [0, 0, 1]]), 10, 20
            )
        with self.assertRaisesRegex(ValueError, "singular or ill-conditioned"):
            extension.transform_landmark_point(np.zeros((3, 3)), 10, 20)

    def test_nonfinite_source_projection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "projection failed"):
            extension.transform_landmark_point(
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, -10]], dtype=float),
                10,
                20,
            )

    def test_projection_domain_detects_horizon_crossing(self):
        matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -640]], dtype=float)
        with self.assertRaisesRegex(ValueError, "horizon crosses"):
            extension.validate_projection_domain(matrix, 1280, 720)

    def test_repeated_landmark_observations_count_once(self):
        base = [{
            "correspondence_id": 1,
            "landmark_id": "L11",
            "landmark_name": "penalty_area_left_sideline_b_corner",
            "reference_frame": 0,
            "image_x": 500.0,
            "image_y": 250.0,
            "pitch_x_m": 16.5,
            "pitch_y_m": 54.16,
        }]
        observations = [
            {
                "landmark_id": "L17",
                "source_frame": frame,
                "reference_frame": 0,
                "reference_x": x,
                "reference_y": y,
            }
            for frame, x, y in ((50, 220.0, 180.0), (150, 221.0, 181.0))
        ]
        correspondences, diagnostics = extension.group_landmark_observations(
            base, observations
        )
        self.assertEqual(len(correspondences), 2)
        self.assertEqual(diagnostics["L17"]["observation_count"], 2)
        self.assertEqual(diagnostics["L17"]["independent_pitch_location_count"], 1)
        self.assertEqual(diagnostics["L17"]["source_frames"], [50, 150])

    def test_repeated_landmark_deviation_is_measured(self):
        rows = [
            {
                "landmark_id": "L17",
                "source_frame": frame,
                "reference_frame": 0,
                "reference_x": x,
                "reference_y": 100.0,
            }
            for frame, x in ((50, 100.0), (150, 140.0))
        ]
        _, diagnostics = extension.group_landmark_observations([], rows)
        self.assertEqual(
            diagnostics["L17"]["maximum_repeat_deviation_px"], 20.0
        )

    def test_unknown_landmark_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown canonical"):
            extension.group_landmark_observations(
                [],
                [{
                    "landmark_id": "L99",
                    "source_frame": 50,
                    "reference_frame": 0,
                    "reference_x": 100.0,
                    "reference_y": 100.0,
                }],
            )

    def test_repeated_same_frame_landmark_is_not_counted_as_new_evidence(self):
        repeated = [
            {
                "landmark_id": "L17",
                "source_frame": 50,
                "reference_frame": 0,
                "reference_x": 100.0,
                "reference_y": 100.0,
            },
            {
                "landmark_id": "L17",
                "source_frame": 50,
                "reference_frame": 0,
                "reference_x": 101.0,
                "reference_y": 101.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "multiple observations from one"):
            extension.group_landmark_observations([], repeated)

    def test_homography_reestimation_supports_held_out_landmark(self):
        geometry = pitch.pitch_geometry()
        image = np.array([
            [80, 70], [1190, 90], [100, 650], [1180, 660], [600, 80], [650, 620]
        ], dtype=float)
        matrix = np.array([
            [105 / 1280, 0, 0],
            [0, 68 / 720, 0],
            [0, 0, 1],
        ], dtype=float)
        target = np.asarray([
            (matrix @ np.array([x, y, 1.0]))[:2] for x, y in image
        ])
        ids = [f"L{i:02d}" for i in range(1, 7)]
        result = pitch.estimate_homography(
            ids, image, target, 1280, 720, geometry
        )
        self.assertTrue(result["valid_for_mapping"], result["errors"])
        self.assertEqual(result["held_out_landmark_id"], ids[-1])
        self.assertLess(result["held_out_error_m"], 1e-6)

    def test_observation_provenance_rejects_other_video_or_segment(self):
        metadata = {
            "tracking_run_id": "run-a",
            "camera_segment_id": "segment_000",
            "reference_frame": 0,
            "source_video_sha256": "video-hash",
            "camera_transform_sha256": "transform-hash",
            "camera_quality_sha256": "quality-hash",
        }
        extension.verify_observation_provenance(
            metadata, "run-a", "segment_000", 0,
            "video-hash", "transform-hash", "quality-hash",
        )
        with self.assertRaisesRegex(ValueError, "camera_segment_id"):
            extension.verify_observation_provenance(
                metadata, "run-a", "segment_001", 0,
                "video-hash", "transform-hash", "quality-hash",
            )
        with self.assertRaisesRegex(ValueError, "source_video_sha256"):
            extension.verify_observation_provenance(
                metadata, "run-a", "segment_000", 0,
                "other-video", "transform-hash", "quality-hash",
            )

    def test_output_protection_refuses_historical_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing = output / "pitch_calibration.json"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                extension._assert_new_output_files(
                    output, ("pitch_calibration.json", "mapping_summary.json")
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_validation_summary_uses_latest_frame_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "validation_summary.csv"
            summary.write_text(
                "test_frame,created_at,max_error_px,median_error_px,p90_error_px\n"
                "50,2026-01-01T00:00:00,20,10,15\n"
                "50,2026-01-02T00:00:00,12,8,11\n",
                encoding="utf-8",
            )
            result = extension._read_validation_summary(summary)
            self.assertEqual(result[50]["max_error_px"], 12)
            self.assertEqual(result[50]["median_error_px"], 8)

    def test_canonical_hull_scenarios_do_not_claim_image_correspondences(self):
        calibration = {
            "pitch_geometry": pitch.pitch_geometry(),
            "pitch_calibration_hull": [
                [16.5, 54.16], [5.5, 43.16], [5.5, 24.84], [16.5, 26.6875]
            ],
        }
        result = extension.theoretical_hull_scenarios(calibration)
        self.assertAlmostEqual(
            result["goal_posts_only"]["canonical_pitch_hull_ratio"], 0.04515, places=4
        )
        self.assertFalse(result["goal_posts_only"]["image_correspondences_assumed"])


if __name__ == "__main__":
    unittest.main()
