import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import dashboard_data as dashboard


RUN_ID = "run-a"
VIDEO = r"C:\project\data\input\sample.mp4"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def track_row(frame=0, track_id="1", run_id=RUN_ID):
    return {
        "run_id": run_id,
        "frame": frame,
        "timestamp": frame / 25,
        "track_id": track_id,
        "class_id": 0,
        "confidence": 0.8,
        "foot_x": 10 + frame,
        "foot_y": 20 + frame,
    }


class DashboardDataTests(unittest.TestCase):
    def test_duration_uses_exclusive_frame_end_and_video_fps(self):
        self.assertEqual(
            dashboard.duration_seconds({
                "start_frame": 0,
                "end_frame_exclusive": 100,
                "video_fps": 25,
            }),
            4.0,
        )
        self.assertEqual(
            dashboard.duration_seconds({
                "start_frame": 10,
                "frames_processed": 50,
                "video_fps": 25,
            }),
            2.0,
        )
        self.assertIsNone(dashboard.duration_seconds({"video_fps": 0}))

    def test_person_detection_count_uses_matched_detection_classes(self):
        detections = pd.DataFrame({"object_class": ["person", " PERSON ", "sports ball"]})
        self.assertEqual(dashboard.person_detection_count(detections), 2)
        self.assertEqual(
            dashboard.person_detection_count(pd.DataFrame({"object_class": []})), 0
        )
        self.assertIsNone(dashboard.person_detection_count(None))
        self.assertIsNone(dashboard.person_detection_count(pd.DataFrame({"frame": []})))

    def test_csv_schema_validation_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            write_csv(path, [{"frame": 1}])
            with self.assertRaisesRegex(dashboard.DataValidationError, "missing required"):
                dashboard.read_csv(path, dashboard.TRACK_COLUMNS, required=True)
            self.assertIsNone(dashboard.read_csv(Path(temporary) / "absent.csv"))
            with self.assertRaises(FileNotFoundError):
                dashboard.read_csv(Path(temporary) / "absent.csv", required=True)

    def test_json_schema_requires_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(dashboard.DataValidationError, "JSON object"):
                dashboard.read_json(path, required=True)

    def test_run_scoped_team_join_and_unknown_classification(self):
        tracks = pd.DataFrame([
            track_row(0, "7", "run-a"),
            track_row(1, "7", "run-a"),
            track_row(0, "7", "run-b"),
        ])
        assignments = pd.DataFrame([{
            "run_id": "run-a",
            "track_id": "7",
            "predicted_team": "team_a",
            "predicted_role": "team_a_outfield",
            "classification_confidence": 0.8,
            "number_of_samples": 4,
        }])
        joined = dashboard._with_track_assignments(tracks, assignments, [])
        self.assertEqual(joined.iloc[0]["predicted_role"], "team_a_outfield")
        self.assertEqual(joined.iloc[2]["predicted_role"], "unknown")

    def test_unique_track_count_is_scoped_to_run_and_person_class(self):
        frame = pd.DataFrame([
            track_row(0, "1"),
            track_row(1, "1"),
            track_row(2, "2"),
            {**track_row(3, "ball"), "class_id": 32},
        ])
        summary = dashboard.track_run_summary(frame)
        self.assertEqual(summary["observation_count"], 3)
        self.assertEqual(summary["unique_track_count"], 2)
        self.assertEqual(summary["frame_count"], 3)
        self.assertEqual(summary["first_frame"], 0)
        self.assertEqual(summary["last_frame"], 2)

    def test_track_confidence_and_gap_summary(self):
        frame = pd.DataFrame([
            track_row(2, "8"),
            {**track_row(5, "8"), "confidence": 0.6},
            track_row(6, "8"),
        ])
        result = dashboard.player_track_analytics(frame, "8")["summary"]
        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["missing_frame_count"], 2)
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(result["largest_gap_frames"], 2)
        self.assertAlmostEqual(result["mean_confidence"], 0.7333333333333333)

    def test_empty_track_inputs_are_safe(self):
        empty = pd.DataFrame(columns=list(dashboard.TRACK_COLUMNS))
        self.assertEqual(dashboard.track_run_summary(empty)["observation_count"], 0)
        self.assertIsNone(dashboard.player_track_analytics(empty, "1")["summary"])

    def test_image_heatmap_counts_finite_samples_without_normalization(self):
        frame = pd.DataFrame({
            "x": [0.0, 0.0, 1.0, np.nan],
            "y": [0.0, 0.0, 1.0, 1.0],
        })
        heat, _, count = dashboard.image_heatmap(
            frame, "x", "y", bins=(2, 2), bounds=(0, 1, 0, 1)
        )
        self.assertEqual(count, 3)
        self.assertEqual(float(heat.sum()), 3.0)
        self.assertEqual(heat[0, 0], 2)
        self.assertEqual(heat[1, 1], 1)

    def test_empty_heatmap_has_expected_shape(self):
        heat, bounds, count = dashboard.image_heatmap(
            pd.DataFrame(), "foot_x", "foot_y", bins=(8, 4)
        )
        self.assertEqual(heat.shape, (4, 8))
        self.assertEqual(float(heat.sum()), 0.0)
        self.assertEqual(count, 0)

    def test_valid_pitch_rows_exclude_invalid_and_nonfinite_positions(self):
        mapping = pd.DataFrame([
            {"pitch_mapping_valid": "True", "pitch_x_m": 20, "pitch_y_m": 30},
            {"pitch_mapping_valid": "False", "pitch_x_m": 25, "pitch_y_m": 35},
            {"pitch_mapping_valid": "True", "pitch_x_m": np.nan, "pitch_y_m": 40},
        ])
        result = dashboard.valid_pitch_rows(mapping)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pitch_x_m"], 20)

    def test_pitch_reason_counts_exclude_valid_rows_and_identify_missing_reason(self):
        mapping = pd.DataFrame([
            {"class_id": 0, "pitch_mapping_valid": "True", "pitch_mapping_invalid_reason": "", "pitch_x_m": 2, "pitch_y_m": 3},
            {"class_id": 0, "pitch_mapping_valid": "False", "pitch_mapping_invalid_reason": "outside_calibration_region", "pitch_x_m": np.nan, "pitch_y_m": np.nan},
            {"class_id": 0, "pitch_mapping_valid": "False", "pitch_mapping_invalid_reason": "", "pitch_x_m": np.nan, "pitch_y_m": np.nan},
            {"class_id": 32, "pitch_mapping_valid": "False", "pitch_mapping_invalid_reason": "untracked_ball", "pitch_x_m": np.nan, "pitch_y_m": np.nan},
        ])
        result = dashboard.pitch_mapping_reason_counts(mapping)
        self.assertEqual(
            dict(zip(result["Reason"], result["Observations"])),
            {"outside_calibration_region": 1, "Not specified": 1},
        )
        self.assertTrue(dashboard.pitch_mapping_reason_counts(None).empty)

    def test_landmark_views_separate_latest_per_frame_from_history(self):
        validation = pd.DataFrame([
            {"run_id": "old-25", "created_at": "2026-10-21T01:00:00+04:00", "test_frame": 25, "median_error_px": 90, "max_error_px": 150},
            {"run_id": "recent-25", "created_at": "2026-10-21T02:00:00+04:00", "test_frame": 25, "median_error_px": 5, "max_error_px": 8},
            {"run_id": "only-50", "created_at": "2026-10-21T01:30:00+04:00", "test_frame": 50, "median_error_px": 12, "max_error_px": 25},
        ])
        latest, history, reviewed_metadata = dashboard.landmark_validation_views(validation)
        self.assertFalse(reviewed_metadata)
        self.assertEqual(latest["run_id"].tolist(), ["recent-25", "only-50"])
        self.assertEqual(len(history), 3)
        self.assertIn("Unreviewed; superseded", history["review_state"].tolist())
        self.assertIn("Latest recorded; unreviewed", latest["review_state"].tolist())

    def test_landmark_views_prioritize_latest_reviewed_when_metadata_exists(self):
        validation = pd.DataFrame([
            {"run_id": "reviewed", "created_at": "2026-10-21T01:00:00+04:00", "test_frame": 25, "median_error_px": 10, "max_error_px": 15, "review_status": "reviewed"},
            {"run_id": "newer-unreviewed", "created_at": "2026-10-21T02:00:00+04:00", "test_frame": 25, "median_error_px": 2, "max_error_px": 3, "review_status": "pending"},
        ])
        latest, history, reviewed_metadata = dashboard.landmark_validation_views(validation)
        self.assertTrue(reviewed_metadata)
        self.assertEqual(latest.iloc[0]["run_id"], "reviewed")
        self.assertEqual(latest.iloc[0]["review_state"], "Latest reviewed")
        self.assertIn(
            "Unreviewed; excluded from reviewed trend",
            history["review_state"].tolist(),
        )

    def test_landmark_views_accept_review_timestamp_metadata(self):
        validation = pd.DataFrame([
            {"run_id": "reviewed", "created_at": "2026-10-21T01:00:00+04:00", "test_frame": 50, "median_error_px": 10, "max_error_px": 15, "reviewed_at": "2026-10-21T03:00:00+04:00"},
            {"run_id": "unreviewed", "created_at": "2026-10-21T02:00:00+04:00", "test_frame": 50, "median_error_px": 2, "max_error_px": 3, "reviewed_at": ""},
        ])
        latest, _, reviewed_metadata = dashboard.landmark_validation_views(validation)
        self.assertTrue(reviewed_metadata)
        self.assertEqual(latest.iloc[0]["run_id"], "reviewed")
        self.assertEqual(latest.iloc[0]["review_state"], "Latest reviewed")

    def test_video_options_prefer_annotated_and_keep_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotated = root / "annotated.mp4"
            original = root / "source.mp4"
            other = root / "other.mp4"
            missing = root / "missing.mp4"
            for path in (annotated, original, other):
                path.touch()
            selected = dashboard.order_video_options(
                [original, other],
                preferred=[annotated, missing],
                original=original,
            )
            self.assertEqual(selected, [annotated.resolve(), other.resolve(), original.resolve()])

    def test_compatible_video_options_map_h264_to_provenance_and_keep_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            source = root / "input" / "match.mp4"
            source.parent.mkdir(parents=True)
            source.touch()
            tracking_dir = output / "tracking_run_a"
            detection_dir = output / "detection"
            team_dir = output / "team_classification"
            compensation_dir = output / "compensation"
            mapping_dir = output / "mapping"
            for directory in (
                tracking_dir, detection_dir, team_dir, compensation_dir, mapping_dir,
                output / "browser_videos",
            ):
                directory.mkdir(parents=True, exist_ok=True)

            tracking_csv = tracking_dir / "tracks_confirmed.csv"
            compensation_csv = compensation_dir / "compensated_tracks.csv"
            tracking_summary = {
                "tracking_run_id": RUN_ID,
                "input_video": str(source),
                "start_frame": 0,
                "end_frame_exclusive": 100,
            }
            detection_summary = {
                "input_video": str(source),
                "start_frame": 0,
                "end_frame_exclusive": 100,
            }
            team_summary = {
                "tracking_run_id": RUN_ID,
                "input_video": str(source),
                "tracking_csv": str(tracking_csv),
                "tracking_frame_range": [0, 100],
            }
            compensation_summary = {
                "tracking_run_id": RUN_ID,
                "compensation_run_id": "comp-a",
                "input_video": str(source),
                "tracking_csv": str(tracking_csv),
                "tracking_frame_range": [0, 100],
            }
            mapping_summary = {
                "tracking_run_id": RUN_ID,
                "compensation_run_id": "comp-a",
                "input_video": str(source),
                "compensated_tracking_csv": str(compensation_csv),
            }
            summary_paths = {
                "tracking": tracking_dir / "tracking_summary.json",
                "detection": detection_dir / "detection_summary.json",
                "teams": team_dir / "team_classification_summary.json",
                "compensation": compensation_dir / "compensation_summary.json",
                "mapping": mapping_dir / "mapping_summary.json",
            }
            for key, summary in (
                ("tracking", tracking_summary),
                ("detection", detection_summary),
                ("teams", team_summary),
                ("compensation", compensation_summary),
                ("mapping", mapping_summary),
            ):
                write_json(summary_paths[key], summary)

            original_names = {
                "detection": (detection_dir, "annotated_detection.mp4"),
                "tracking": (tracking_dir, "annotated_tracking.mp4"),
                "teams": (team_dir, "annotated_teams.mp4"),
                "compensation": (compensation_dir, "trajectory_comparison.mp4"),
                "mapping": (mapping_dir, "annotated_pitch_mapping.mp4"),
            }
            for directory, filename in original_names.values():
                (directory / filename).touch()
            h264_names = [
                "annotated_detection_h264.mp4",
                "annotated_tracking_h264.mp4",
                "annotated_teams_h264.mp4",
                "trajectory_comparison_h264.mp4",
                "annotated_pitch_mapping_h264.mp4",
            ]
            for filename in h264_names:
                (output / "browser_videos" / filename).touch()

            dataset = {
                "run_id": RUN_ID,
                "video_path": source,
                "tracking_dir": tracking_dir,
                "tracking_summary": tracking_summary,
                "detection_summary": detection_summary,
                "team_summary": team_summary,
                "compensation_summary": compensation_summary,
                "mapping_summary": mapping_summary,
                "source_summaries": {
                    key: str(path) for key, path in summary_paths.items()
                },
            }
            options = dashboard.compatible_video_options(output, dataset)
            self.assertEqual(
                [option["label"] for option in options],
                [
                    "YOLO detection · H.264",
                    "ByteTrack tracking · H.264",
                    "Team classification · H.264",
                    "Camera-compensation comparison · H.264",
                    "Pitch mapping · H.264",
                    "YOLO detection · original",
                    "ByteTrack tracking · original",
                    "Team classification · original",
                    "Camera-compensation comparison · original",
                    "Pitch mapping · original",
                    "Source video · match.mp4",
                ],
            )
            self.assertEqual(
                options[2]["path"],
                (output / "browser_videos" / "annotated_teams_h264.mp4").resolve(),
            )

    def test_video_options_fall_back_when_h264_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            source = root / "source.mp4"
            source.touch()
            tracking_dir = output / "tracking"
            tracking_dir.mkdir(parents=True)
            summary_path = tracking_dir / "tracking_summary.json"
            summary = {
                "tracking_run_id": RUN_ID,
                "input_video": str(source),
                "start_frame": 0,
                "end_frame_exclusive": 10,
            }
            write_json(summary_path, summary)
            original = tracking_dir / "annotated_tracking.mp4"
            original.touch()
            dataset = {
                "run_id": RUN_ID,
                "video_path": source,
                "tracking_dir": tracking_dir,
                "tracking_summary": summary,
                "source_summaries": {"tracking": str(summary_path)},
            }

            options = dashboard.compatible_video_options(output, dataset)
            self.assertEqual(
                [option["label"] for option in options],
                ["ByteTrack tracking · original", "Source video · source.mp4"],
            )
            self.assertEqual(options[0]["path"], original.resolve())

    def test_video_options_exclude_derived_artifacts_from_other_tracking_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            source = root / "source.mp4"
            source.touch()
            tracking_dir = output / "tracking_run_b"
            team_dir = output / "team_classification"
            compensation_dir = output / "compensation"
            mapping_dir = output / "mapping"
            for directory in (tracking_dir, team_dir, compensation_dir, mapping_dir):
                directory.mkdir(parents=True, exist_ok=True)

            selected_tracking = {
                "tracking_run_id": "run-b",
                "input_video": str(source),
                "start_frame": 0,
                "end_frame_exclusive": 100,
            }
            old_team = {
                "tracking_run_id": RUN_ID,
                "input_video": str(source),
                "tracking_csv": str(root / "old_tracks.csv"),
                "tracking_frame_range": [0, 100],
            }
            old_compensation = {
                "tracking_run_id": RUN_ID,
                "compensation_run_id": "comp-a",
                "input_video": str(source),
                "tracking_csv": str(root / "old_tracks.csv"),
                "tracking_frame_range": [0, 100],
            }
            old_mapping = {
                "tracking_run_id": RUN_ID,
                "compensation_run_id": "comp-a",
                "input_video": str(source),
                "compensated_tracking_csv": str(compensation_dir / "compensated_tracks.csv"),
            }
            paths = {
                "tracking": tracking_dir / "tracking_summary.json",
                "teams": team_dir / "team_classification_summary.json",
                "compensation": compensation_dir / "compensation_summary.json",
                "mapping": mapping_dir / "mapping_summary.json",
            }
            for key, value in (
                ("tracking", selected_tracking),
                ("teams", old_team),
                ("compensation", old_compensation),
                ("mapping", old_mapping),
            ):
                write_json(paths[key], value)
            (tracking_dir / "annotated_tracking.mp4").touch()
            for name in (
                "annotated_tracking_h264.mp4",
                "annotated_teams_h264.mp4",
                "trajectory_comparison_h264.mp4",
                "annotated_pitch_mapping_h264.mp4",
            ):
                browser_dir = output / "browser_videos"
                browser_dir.mkdir(parents=True, exist_ok=True)
                (browser_dir / name).touch()
            (team_dir / "annotated_teams.mp4").touch()
            (compensation_dir / "trajectory_comparison.mp4").touch()
            (mapping_dir / "annotated_pitch_mapping.mp4").touch()

            dataset = {
                "run_id": "run-b",
                "video_path": source,
                "tracking_dir": tracking_dir,
                "tracking_summary": selected_tracking,
                "team_summary": old_team,
                "compensation_summary": old_compensation,
                "mapping_summary": old_mapping,
                "source_summaries": {key: str(path) for key, path in paths.items()},
            }
            labels = [
                option["label"]
                for option in dashboard.compatible_video_options(output, dataset)
            ]
            self.assertEqual(
                labels,
                ["ByteTrack tracking · H.264", "ByteTrack tracking · original", "Source video · source.mp4"],
            )

    def test_camera_compensated_heatmap_data_is_segment_isolated(self):
        compensated = pd.DataFrame([
            {"camera_segment_id": "segment_000", "compensation_valid": "True", "compensated_x": 2, "compensated_y": 3},
            {"camera_segment_id": "segment_001", "compensation_valid": "True", "compensated_x": 9, "compensated_y": 9},
            {"camera_segment_id": "segment_000", "compensation_valid": "False", "compensated_x": 4, "compensated_y": 5},
        ])
        result = dashboard.camera_segment_rows(compensated, "segment_000")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["compensated_x"], 2)

    def test_environment_paths_are_project_relative_by_default(self):
        paths = dashboard.ProjectPaths.from_environment(Path(__file__).resolve().parents[1])
        self.assertEqual(paths.output_dir, Path(__file__).resolve().parents[1] / "data" / "output")

    def test_dataset_loading_joins_compatible_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            video = Path(temporary) / "sample.mp4"
            tracking_dir = output / "tracking_run"
            write_json(tracking_dir / "tracking_summary.json", {
                "tracking_run_id": RUN_ID,
                "input_video": str(video),
                "video_fps": 25,
                "source_frame_count": 100,
                "start_frame": 0,
                "end_frame_exclusive": 100,
                "frames_processed": 100,
            })
            tracks = [track_row(0), track_row(1)]
            write_csv(tracking_dir / "tracks_confirmed.csv", tracks)
            write_csv(tracking_dir / "tracks.csv", tracks)
            team_dir = output / "team_classification"
            write_json(team_dir / "team_classification_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": str(video),
                "tracking_csv": str(tracking_dir / "tracks_confirmed.csv"),
            })
            assignment = {
                "run_id": RUN_ID, "track_id": "1",
                "predicted_team": "team_a", "predicted_role": "team_a_outfield",
                "classification_confidence": 0.9, "number_of_samples": 5,
            }
            write_csv(team_dir / "team_assignments.csv", [assignment])
            comp_dir = output / "compensated_tracking"
            write_json(comp_dir / "compensation_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": str(video),
                "compensation_run_id": "comp-a",
                "tracking_csv": str(tracking_dir / "tracks_confirmed.csv"),
            })
            comp_rows = [{
                **track_row(0),
                "camera_segment_id": "segment_000",
                "reference_frame": 0,
                "compensation_valid": "True",
                "compensated_x": 10,
                "compensated_y": 20,
            }]
            write_csv(comp_dir / "compensated_tracks.csv", comp_rows)
            map_dir = output / "pitch_mapping" / "mapping"
            write_json(map_dir / "mapping_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": str(video),
                "compensation_run_id": "comp-a",
                "compensated_tracking_csv": str(comp_dir / "compensated_tracks.csv"),
            })
            mapping_row = {
                **comp_rows[0],
                "pitch_x_m": 2,
                "pitch_y_m": 3,
                "pitch_mapping_valid": "True",
                "pitch_mapping_invalid_reason": "",
            }
            write_csv(map_dir / "mapped_player_positions.csv", [mapping_row])
            detection_row = {
                "frame_number": 0,
                "object_class": "person",
                "confidence": 0.8,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
            }
            for directory, end_frame, detections in (
                ("detection", 100, [detection_row]),
                ("detection_full", 1049, [detection_row, {
                    **detection_row, "object_class": "sports ball",
                }]),
            ):
                detection_dir = output / directory
                write_json(detection_dir / "detection_summary.json", {
                    "input_video": str(video),
                    "start_frame": 0,
                    "end_frame_exclusive": end_frame,
                    "frames_processed": end_frame,
                })
                write_csv(detection_dir / "detections.csv", detections)
            dataset = dashboard.load_dashboard_dataset(output, tracking_dir)
            self.assertEqual(dataset["run_id"], RUN_ID)
            self.assertEqual(len(dataset["confirmed_tracks"]), 2)
            self.assertEqual(dataset["confirmed_tracks"].iloc[0]["predicted_role"], "team_a_outfield")
            self.assertEqual(dataset["compensated_tracks"].iloc[0]["camera_segment_id"], "segment_000")
            self.assertEqual(len(dashboard.valid_pitch_rows(dataset["mapping"])), 1)
            self.assertEqual(dashboard.person_detection_count(dataset["detections"]), 1)
            self.assertEqual(dataset["detection_summary"]["end_frame_exclusive"], 100)
            self.assertEqual(dataset["errors"], [])

    def test_dataset_rejects_mixed_tracking_run_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            tracking_dir = output / "tracking_run"
            write_json(tracking_dir / "tracking_summary.json", {
                "tracking_run_id": RUN_ID,
                "input_video": VIDEO,
                "video_fps": 25,
            })
            write_csv(tracking_dir / "tracks_confirmed.csv", [track_row(run_id="run-b")])
            with self.assertRaisesRegex(dashboard.DataValidationError, "selected tracking run UUID"):
                dashboard.load_dashboard_dataset(output, tracking_dir)

    def test_incompatible_mapping_compensation_run_is_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            tracking_dir = output / "tracking_run"
            write_json(tracking_dir / "tracking_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": VIDEO, "video_fps": 25,
            })
            row = track_row()
            write_csv(tracking_dir / "tracks_confirmed.csv", [row])
            write_csv(tracking_dir / "tracks.csv", [row])
            comp_dir = output / "comp"
            write_json(comp_dir / "compensation_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": VIDEO, "compensation_run_id": "comp-a",
                "tracking_csv": str(tracking_dir / "tracks_confirmed.csv"),
            })
            comp_row = {
                **row, "camera_segment_id": "segment_000", "reference_frame": 0,
                "compensation_valid": "True", "compensated_x": 1, "compensated_y": 2,
            }
            write_csv(comp_dir / "compensated_tracks.csv", [comp_row])
            map_dir = output / "pitch_mapping" / "mapping"
            write_json(map_dir / "mapping_summary.json", {
                "tracking_run_id": RUN_ID, "input_video": VIDEO, "compensation_run_id": "other",
                "compensated_tracking_csv": str(comp_dir / "compensated_tracks.csv"),
            })
            mapping_row = {
                **comp_row, "pitch_x_m": 1, "pitch_y_m": 2,
                "pitch_mapping_valid": "True", "pitch_mapping_invalid_reason": "",
            }
            write_csv(map_dir / "mapped_player_positions.csv", [mapping_row])
            dataset = dashboard.load_dashboard_dataset(output, tracking_dir)
            self.assertIsNone(dataset["mapping"])
            self.assertTrue(any("different run IDs" in error for error in dataset["errors"]))


if __name__ == "__main__":
    unittest.main()
