import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import compensate_player_tracks as compensation


def quality_rows(count, failed=()):
    failed = set(failed)
    rows = []
    for frame in range(count):
        is_failed = frame in failed
        rows.append({
            "frame": frame,
            "method": "failed" if is_failed else ("reference" if frame == 0 else "flow"),
            "valid": not is_failed,
        })
    return rows


def tracking_row(frame, track_id="12", run_id="run-a", class_id=0, foot=(10, 20)):
    return {
        "run_id": run_id,
        "frame": str(frame),
        "timestamp": f"{frame / 25:.2f}",
        "track_id": track_id,
        "class_id": str(class_id),
        "confidence": "0.9",
        "x1": "5",
        "y1": "5",
        "x2": "15",
        "y2": "25",
        "center_x": "10",
        "center_y": "15",
        "foot_x": str(foot[0]) if foot else "",
        "foot_y": str(foot[1]) if foot else "",
    }


class CompensatePlayerTracksTests(unittest.TestCase):
    def test_identity_transform_preserves_point(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        frames = compensation.build_frame_transforms(matrices, quality_rows(2))
        projected, error = compensation.project_point(frames[1].matrix, 123.5, 45.25)
        self.assertEqual(error, "")
        self.assertEqual(projected, (123.5, 45.25))

    def test_translation_only_transform(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        matrices[1] = np.array([
            [1, 0, -12],
            [0, 1, 7],
            [0, 0, 1],
        ], dtype=np.float64)
        frames = compensation.build_frame_transforms(matrices, quality_rows(2))
        projected, error = compensation.project_point(frames[1].matrix, 20, 30)
        self.assertEqual(error, "")
        self.assertEqual(projected, (8.0, 37.0))

    def test_noncommuting_scale_translation_composition_order(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        matrices[1] = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        matrices[2] = np.array([
            [1, 0, 3],
            [0, 1, 4],
            [0, 0, 1],
        ], dtype=np.float64)
        frames = compensation.build_frame_transforms(matrices, quality_rows(3))
        projected, error = compensation.project_point(frames[2].matrix, 1, 2)
        self.assertEqual(error, "")
        self.assertEqual(projected, (8.0, 12.0))
        self.assertTrue(np.allclose(frames[2].matrix, matrices[1] @ matrices[2]))
        self.assertFalse(np.allclose(frames[2].matrix, matrices[2] @ matrices[1]))

    def test_person_foot_point_projection_and_metadata_preservation(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        matrices[1, 0, 2] = -5
        frames = compensation.build_frame_transforms(matrices, quality_rows(2))
        source = tracking_row(1, foot=(10, 20))
        fields, rows = compensation.compensate_rows(
            [source], list(source), frames
        )
        row = rows[0]
        self.assertEqual(row["frame"], "1")
        self.assertEqual(row["timestamp"], "0.04")
        self.assertEqual(row["run_id"], "run-a")
        self.assertEqual(row["track_id"], "12")
        self.assertEqual(row["foot_x"], "10")
        self.assertEqual(row["foot_y"], "20")
        self.assertEqual(row["compensated_x"], 5.0)
        self.assertEqual(row["compensated_y"], 20.0)
        self.assertEqual(row["compensation_valid"], "True")
        self.assertIn("camera_segment_id", fields)

    def test_noncommuting_composition_for_rows_is_correct(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        matrices[1] = np.diag([2.0, 2.0, 1.0])
        matrices[2] = np.array([
            [1, 0, 3],
            [0, 1, 4],
            [0, 0, 1],
        ], dtype=np.float64)
        frames = compensation.build_frame_transforms(matrices, quality_rows(3))
        row = tracking_row(2, foot=(1, 2))
        _, results = compensation.compensate_rows([row], list(row), frames)
        self.assertEqual(
            (results[0]["compensated_x"], results[0]["compensated_y"]),
            (8.0, 12.0),
        )

    def test_failed_transition_starts_new_reference_without_bridging(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 5, axis=0)
        matrices[2] = np.full((3, 3), np.nan)
        matrices[2, 0, 0] = np.nan
        matrices[3, 0, 2] = 100
        frames = compensation.build_frame_transforms(
            matrices, quality_rows(5, failed={2})
        )
        self.assertEqual(frames[1].segment_id, "segment_000")
        self.assertEqual(frames[2].segment_id, "segment_001")
        self.assertEqual(frames[2].reference_frame, 2)
        self.assertEqual(frames[2].segment_start_reason, "failed_transition_new_reference")
        self.assertEqual(frames[3].segment_id, "segment_001")
        projected, error = compensation.project_point(frames[3].matrix, 1, 2)
        self.assertEqual(error, "")
        self.assertEqual(projected, (101.0, 2.0))

    def test_consecutive_failed_transitions_create_unjoined_segments(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 6, axis=0)
        matrices[2:5] = np.nan
        frames = compensation.build_frame_transforms(
            matrices, quality_rows(6, failed={2, 3, 4})
        )
        self.assertEqual(
            [frame.segment_id for frame in frames],
            [
                "segment_000",
                "segment_000",
                "segment_001",
                "segment_002",
                "segment_003",
                "segment_003",
            ],
        )
        self.assertEqual(frames[5].reference_frame, 4)

    def test_nonfinite_matrix_marked_invalid_then_next_frame_references_anew(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
        matrices[1, 0, 0] = np.nan
        frames = compensation.build_frame_transforms(matrices, quality_rows(4))
        self.assertFalse(frames[1].valid)
        self.assertEqual(frames[1].invalid_reason, "nonfinite_transform")
        self.assertEqual(frames[2].segment_id, "segment_001")
        self.assertEqual(frames[2].reference_frame, 2)
        rows = [tracking_row(1), tracking_row(2)]
        _, results = compensation.compensate_rows(
            rows, list(rows[0]), frames
        )
        self.assertEqual(results[0]["compensation_valid"], "False")
        self.assertEqual(results[0]["invalid_reason"], "nonfinite_transform")
        self.assertEqual(results[1]["compensation_valid"], "True")

    def test_singular_and_ill_conditioned_matrices_are_rejected(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        matrices[1] = np.diag([1.0, 1.0, 0.0])
        frames = compensation.build_frame_transforms(matrices, quality_rows(2))
        self.assertFalse(frames[1].valid)
        self.assertEqual(frames[1].invalid_reason, "singular_transform")

        matrices[1] = np.diag([1.0, 1.0, 1e-14])
        frames = compensation.build_frame_transforms(
            matrices, quality_rows(2), condition_limit=1e10
        )
        self.assertFalse(frames[1].valid)
        self.assertEqual(frames[1].invalid_reason, "ill_conditioned_transform")

    def test_near_zero_homogeneous_denominator_rejected(self):
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1e-14],
        ], dtype=np.float64)
        projected, error = compensation.project_point(matrix, 5, 8)
        self.assertIsNone(projected)
        self.assertEqual(error, "near_zero_homogeneous_denominator")

    def test_empty_tracking_input_remains_empty(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        frames = compensation.build_frame_transforms(matrices, quality_rows(2))
        fields, rows = compensation.compensate_rows(
            [], ["run_id", "frame", "timestamp", "track_id", "class_id", "foot_x", "foot_y"],
            frames,
        )
        self.assertEqual(rows, [])
        self.assertIn("compensated_x", fields)

    def test_non_person_rows_preserved_but_not_projected(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 1, axis=0)
        frames = compensation.build_frame_transforms(matrices, quality_rows(1))
        source = tracking_row(0, track_id="", class_id=32, foot=None)
        _, rows = compensation.compensate_rows([source], list(source), frames)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["compensation_valid"], "False")
        self.assertEqual(rows[0]["invalid_reason"], "not_person_detection")
        self.assertEqual(rows[0]["track_id"], "")
        self.assertEqual(rows[0]["run_id"], "run-a")

    def test_frame_number_track_id_and_run_id_preserved(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
        frames = compensation.build_frame_transforms(matrices, quality_rows(8))
        source = tracking_row(7, track_id="55", run_id="unique-run")
        _, results = compensation.compensate_rows([source], list(source), frames)
        self.assertEqual(results[0]["frame"], "7")
        self.assertEqual(results[0]["track_id"], "55")
        self.assertEqual(results[0]["run_id"], "unique-run")
        self.assertEqual(results[0]["reference_frame"], 0)

    def test_mixed_run_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracks.csv"
            first = tracking_row(0, run_id="run-a")
            second = tracking_row(1, run_id="run-b")
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(first))
                writer.writeheader()
                writer.writerows([first, second])
            with self.assertRaisesRegex(ValueError, "Run ID mismatch"):
                compensation.read_tracking_rows(path, "run-a", 2)

    def test_tracking_rows_must_match_declared_run_range_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracks.csv"
            row = tracking_row(4, run_id="run-a")
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            _, rows = compensation.read_tracking_rows(
                path, "run-a", 8, start_frame=4, end_frame=5, expected_fps=25
            )
            self.assertEqual(rows[0]["frame"], "4")
            with self.assertRaisesRegex(ValueError, "outside the run range"):
                compensation.read_tracking_rows(
                    path, "run-a", 8, start_frame=5, end_frame=6, expected_fps=25
                )
            row["timestamp"] = "0.5"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Timestamp"):
                compensation.read_tracking_rows(
                    path, "run-a", 8, start_frame=4, end_frame=5, expected_fps=25
                )

    def test_invalid_transition_observation_is_a_new_segment_reference(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        matrices[1] = np.full((3, 3), np.nan)
        frames = compensation.build_frame_transforms(
            matrices, quality_rows(3, failed={1})
        )
        row = tracking_row(1, foot=(10, 20))
        _, results = compensation.compensate_rows([row], list(row), frames)
        self.assertEqual(results[0]["compensation_valid"], "True")
        self.assertEqual(results[0]["camera_segment_id"], "segment_001")
        self.assertEqual(results[0]["reference_frame"], 1)
        self.assertEqual(results[0]["segment_start_reason"], "failed_transition_new_reference")
        self.assertEqual((results[0]["compensated_x"], results[0]["compensated_y"]), (10.0, 20.0))

    def test_unique_output_paths_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            outputs, temporary = compensation.create_output_paths(directory)
            for path in temporary.values():
                path.unlink(missing_ok=True)
            self.assertEqual(set(outputs), set(compensation.OUTPUT_NAMES))
            outputs["compensated_tracks.csv"].write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                compensation.create_output_paths(directory)

    def test_summary_counts_valid_invalid_and_segments(self):
        matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
        matrices[2] = np.nan
        frames = compensation.build_frame_transforms(
            matrices, quality_rows(4, failed={2})
        )
        source = [tracking_row(1), tracking_row(2, class_id=32, foot=None)]
        _, rows = compensation.compensate_rows(source, list(source[0]), frames)
        result = compensation.summarize_compensation(rows, frames, 1, 4)
        self.assertEqual(result["person_observations"], 1)
        self.assertEqual(result["valid_compensated_observations"], 1)
        self.assertEqual(result["invalid_compensated_observations"], 0)
        self.assertEqual(result["camera_segments_encountered"], 2)
        self.assertEqual(result["failed_transition_segment_starts_in_run"], [2])
        self.assertEqual(result["maximum_absolute_compensated_coordinate"], 20.0)
        self.assertIn("not directly comparable", result["other_segments_coordinate_comparability"])

    def test_tracking_summary_loader_requires_explicit_matching_set_and_run(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            video = directory / "sample.mp4"
            video.write_bytes(b"video placeholder")
            tracks = directory / "tracks_confirmed.csv"
            tracks.write_text("header\n", encoding="utf-8")
            (directory / "tracking_summary.json").write_text(json.dumps({
                "tracking_run_id": "run-a",
                "input_video": str(video),
                "start_frame": 4,
                "end_frame_exclusive": 5,
                "video_fps": 25,
                "frame_width": 1280,
                "frame_height": 720,
                "raw_tracks_csv": "tracks.csv",
                "confirmed_candidates_csv": "tracks_confirmed.csv",
            }), encoding="utf-8")
            result = compensation._load_tracking_metadata(
                tracks, "confirmed"
            )
            self.assertEqual(result[1], "run-a")
            self.assertEqual((result[3], result[4]), (4, 5))
            with self.assertRaisesRegex(ValueError, "track-set raw"):
                compensation._load_tracking_metadata(tracks, "raw")


if __name__ == "__main__":
    unittest.main()
