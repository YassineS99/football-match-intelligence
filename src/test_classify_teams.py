import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_teams as teams


def solid_frame(color, shape=(60, 50, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:] = color
    return frame


def tracking_row(frame=0, track_id="1", run_id="run-a", box=(10, 5, 30, 55)):
    x1, y1, x2, y2 = box
    return {
        "run_id": run_id,
        "frame": frame,
        "timestamp": frame / 25,
        "track_id": track_id,
        "class_id": 0,
        "confidence": "0.9",
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
    }


class ClassifyTeamsTests(unittest.TestCase):
    def test_upper_body_crop_selects_center_torso_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = teams.upper_body_crop(frame, (10, 10, 50, 90))
        self.assertEqual(crop.shape[:2], (30, 24))
        self.assertTrue(np.shares_memory(frame, crop))

    def test_out_of_bounds_bounding_box_is_clipped(self):
        frame = solid_frame((0, 0, 200), shape=(25, 20, 3))
        crop = teams.upper_body_crop(frame, (-10, -10, 10, 30))
        self.assertIsNotNone(crop)
        self.assertGreater(crop.size, 0)
        self.assertLessEqual(crop.shape[0], frame.shape[0])
        self.assertLessEqual(crop.shape[1], frame.shape[1])
        self.assertIsNone(teams.upper_body_crop(frame, (30, 1, 40, 15)))

    def test_empty_and_occluded_crops_are_rejected(self):
        frame = solid_frame((80, 80, 80))
        self.assertIsNone(teams.upper_body_color_features(frame, (5, 5, 35, 55)))
        self.assertIsNone(teams.upper_body_color_features(frame, (5, 5, 5, 55)))
        self.assertIsNone(teams.upper_body_color_features(frame, (np.nan, 1, 4, 5)))

    def test_red_and_blue_jersey_samples_are_separated(self):
        box = (10, 5, 30, 55)
        red = teams.upper_body_color_features(solid_frame((0, 0, 210)), box)
        blue = teams.upper_body_color_features(solid_frame((210, 60, 0)), box)
        red_assignment = teams.classify_track(red, 8, 3, 0.24, 0.10)
        blue_assignment = teams.classify_track(blue, 8, 3, 0.24, 0.10)
        self.assertEqual(red_assignment[:2], ("team_a", "team_a_outfield"))
        self.assertEqual(blue_assignment[:2], ("team_b", "team_b_outfield"))
        self.assertGreater(red_assignment[2], 0)
        self.assertGreater(blue_assignment[2], 0)

    def test_track_feature_aggregation_resists_single_outlier(self):
        red = {
            "team_a_red_share": 1.0,
            "team_b_blue_share": 0.0,
            "referee_yellow_share": 0.0,
            "goalkeeper_green_share": 0.0,
        }
        blue_outlier = {
            "team_a_red_share": 0.0,
            "team_b_blue_share": 1.0,
            "referee_yellow_share": 0.0,
            "goalkeeper_green_share": 0.0,
        }
        aggregated = teams.aggregate_track_features(
            {("run-a", "1"): [red, red, blue_outlier, red, red]}
        )
        self.assertEqual(aggregated[("run-a", "1")]["team_a_red_share"], 1.0)
        result = teams.classify_track(
            aggregated[("run-a", "1")], 5, 3, 0.24, 0.10
        )
        self.assertEqual(result[:2], ("team_a", "team_a_outfield"))

    def test_yellow_and_green_are_provisional_role_candidates(self):
        box = (10, 5, 30, 55)
        yellow = teams.upper_body_color_features(solid_frame((0, 210, 255)), box)
        green = teams.upper_body_color_features(solid_frame((0, 220, 0)), box)
        self.assertEqual(
            teams.classify_track(yellow, 8, 3, 0.24, 0.10)[:2],
            ("unknown", "referee_candidate"),
        )
        self.assertEqual(
            teams.classify_track(green, 8, 3, 0.24, 0.10)[:2],
            ("unknown", "goalkeeper_candidate"),
        )
        self.assertEqual(
            teams.classify_track(yellow, 4, 3, 0.24, 0.10)[:2],
            ("unknown", "unknown"),
        )

    def test_green_contamination_alone_does_not_create_goalkeeper_candidate(self):
        mixed = {
            "team_a_red_share": 0.02,
            "team_b_blue_share": 0.34,
            "referee_yellow_share": 0.03,
            "goalkeeper_green_share": 0.50,
        }
        self.assertEqual(
            teams.classify_track(mixed, 8, 3, 0.24, 0.10)[:2],
            ("unknown", "unknown"),
        )

    def test_low_evidence_and_ambiguous_colors_stay_unknown(self):
        balanced = {
            "team_a_red_share": 0.31,
            "team_b_blue_share": 0.30,
            "referee_yellow_share": 0.0,
            "goalkeeper_green_share": 0.0,
        }
        self.assertEqual(
            teams.classify_track(balanced, 6, 3, 0.24, 0.10),
            ("unknown", "unknown", 0.0),
        )
        self.assertEqual(
            teams.classify_track(balanced, 2, 3, 0.24, 0.10),
            ("unknown", "unknown", 0.0),
        )

    def test_assignments_are_scoped_by_tracking_run_and_track_id(self):
        rows = [
            tracking_row(0, "1", "run-a"),
            tracking_row(0, "1", "run-b"),
        ]
        frame = solid_frame((0, 0, 210))
        assignments = teams._classify_tracks(
            rows, {0: frame}, 1, 1, 0.24, 0.10
        )
        self.assertEqual(set(assignments), {("run-a", "1"), ("run-b", "1")})
        self.assertEqual(assignments[("run-a", "1")]["run_id"], "run-a")
        self.assertEqual(assignments[("run-b", "1")]["run_id"], "run-b")

    def test_csv_output_schema_is_explicit(self):
        self.assertEqual(
            teams.ASSIGNMENT_FIELDS,
            [
                "run_id", "track_id", "predicted_team", "predicted_role",
                "classification_confidence", "number_of_samples",
                "classification_method",
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assignments.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=teams.ASSIGNMENT_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "run_id": "run-a",
                    "track_id": "1",
                    "predicted_team": "team_a",
                    "predicted_role": "team_a_outfield",
                    "classification_confidence": 0.9,
                    "number_of_samples": 5,
                    "classification_method": teams.CLASSIFICATION_METHOD,
                })
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, teams.ASSIGNMENT_FIELDS)
                self.assertEqual(next(reader)["run_id"], "run-a")

    def test_empty_tracking_input_produces_no_assignments(self):
        self.assertEqual(
            teams._classify_tracks([], {}, 3, 3, 0.24, 0.10),
            {},
        )

    def test_generated_output_schemas_and_video_are_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.full((48, 64, 3), (0, 0, 190), dtype=np.uint8))
            writer.release()
            row = tracking_row(0, "7", "run-a", (5, 3, 20, 43))
            assignment = {
                "run_id": "run-a",
                "track_id": "7",
                "predicted_team": "team_a",
                "predicted_role": "team_a_outfield",
                "classification_confidence": 0.9,
                "number_of_samples": 1,
                "classification_method": teams.CLASSIFICATION_METHOD,
            }
            summary = teams._write_outputs(
                root / "output",
                [row],
                {("run-a", "7"): assignment},
                source,
                0,
                1,
                25.0,
                64,
                48,
                1,
                "run-a",
                {
                    "tracking_csv": "tracks_confirmed.csv",
                    "minimum_samples": 1,
                    "minimum_color_share": 0.24,
                    "minimum_color_margin": 0.1,
                    "source_person_observations": 1,
                },
            )
            output = root / "output"
            with (output / "team_assignments.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, teams.ASSIGNMENT_FIELDS)
                self.assertEqual(next(reader)["track_id"], "7")
            saved_summary = json.loads(
                (output / "team_classification_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved_summary["tracking_run_id"], "run-a")
            self.assertEqual(summary["tracks"], 1)
            self.assertIsNotNone(
                cv2.imread(str(output / "team_classification_preview.png"))
            )
            video = cv2.VideoCapture(str(output / "annotated_teams.mp4"))
            self.assertTrue(video.isOpened())
            ok, frame = video.read()
            video.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (48, 64))

    def test_existing_historical_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            protected = output_dir / "team_assignments.csv"
            protected.write_text("historical", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                teams._write_outputs(
                    output_dir,
                    [],
                    {},
                    Path(temp) / "nonexistent.mp4",
                    0,
                    1,
                    25.0,
                    50,
                    60,
                    1,
                    "run-a",
                    {"source_person_observations": 0},
                )
            self.assertEqual(protected.read_text(encoding="utf-8"), "historical")

    def test_cli_rejects_invalid_sampling_and_confidence_parameters(self):
        parser = teams.build_parser()
        args = parser.parse_args(["--sample-every", "0"])
        with self.assertRaisesRegex(ValueError, "--sample-every"):
            teams.validate_args(args)
        args = parser.parse_args(["--min-color-share", "1.5"])
        with self.assertRaisesRegex(ValueError, "--min-color-share"):
            teams.validate_args(args)


if __name__ == "__main__":
    unittest.main()
