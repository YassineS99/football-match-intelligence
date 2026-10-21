import ast
import csv
import hashlib
import json
import os
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


def make_video_fixture(root, run_id=RUN_ID, frame_range=(0, 100), source=None):
    root = Path(root)
    output = root / "output"
    start_frame, end_frame = frame_range
    if source is None:
        source = root / "input" / "match.mp4"
    source = Path(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_bytes(b"source footage fixture")

    tracking_dir = output / f"tracking_{run_id}"
    detection_dir = output / f"detection_{run_id}"
    team_dir = output / f"teams_{run_id}"
    compensation_dir = output / f"compensation_{run_id}"
    mapping_dir = output / f"mapping_{run_id}"
    tracking_csv = tracking_dir / "tracks_confirmed.csv"
    compensation_csv = compensation_dir / "compensated_tracks.csv"
    tracking_summary = {
        "tracking_run_id": run_id,
        "input_video": str(source),
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
    }
    detection_summary = {
        "input_video": str(source),
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
    }
    team_summary = {
        "tracking_run_id": run_id,
        "input_video": str(source),
        "tracking_csv": str(tracking_csv),
        "tracking_frame_range": [start_frame, end_frame],
    }
    compensation_summary = {
        "tracking_run_id": run_id,
        "compensation_run_id": f"comp-{run_id}",
        "input_video": str(source),
        "tracking_csv": str(tracking_csv),
        "tracking_frame_range": [start_frame, end_frame],
    }
    mapping_summary = {
        "tracking_run_id": run_id,
        "compensation_run_id": f"comp-{run_id}",
        "input_video": str(source),
        "compensated_tracking_csv": str(compensation_csv),
    }
    summaries = {
        "tracking": (tracking_dir / "tracking_summary.json", tracking_summary),
        "detection": (detection_dir / "detection_summary.json", detection_summary),
        "teams": (team_dir / "team_classification_summary.json", team_summary),
        "compensation": (
            compensation_dir / "compensation_summary.json",
            compensation_summary,
        ),
        "mapping": (mapping_dir / "mapping_summary.json", mapping_summary),
    }
    for path, summary in summaries.values():
        write_json(path, summary)

    definitions = {
        "detection": (
            "detection",
            detection_dir / "annotated_detection.mp4",
            "annotated_detection_h264.mp4",
        ),
        "tracking": (
            "tracking",
            tracking_dir / "annotated_tracking.mp4",
            "annotated_tracking_h264.mp4",
        ),
        "teams": (
            "team_classification",
            team_dir / "annotated_teams.mp4",
            "annotated_teams_h264.mp4",
        ),
        "compensation": (
            "camera_compensation",
            compensation_dir / "trajectory_comparison.mp4",
            "trajectory_comparison_h264.mp4",
        ),
        "mapping": (
            "pitch_mapping",
            mapping_dir / "annotated_pitch_mapping.mp4",
            "annotated_pitch_mapping_h264.mp4",
        ),
    }
    variants = {}
    for key, (analysis_type, original, filename) in definitions.items():
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(f"{key}:{run_id}:{frame_range}".encode("utf-8"))
        h264 = output / "browser_videos" / run_id / filename
        h264.parent.mkdir(parents=True, exist_ok=True)
        h264.write_bytes(f"H264:{key}:{run_id}:{frame_range}".encode("utf-8"))
        provenance = {
            "tracking_run_id": run_id,
            "analysis_type": analysis_type,
            "original_video_path": str(original.resolve()),
            "original_video_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "source_footage_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "conversion_output_path": (
                Path("browser_videos") / run_id / filename
            ).as_posix(),
            "conversion_output_sha256": hashlib.sha256(h264.read_bytes()).hexdigest(),
            "conversion_output_size_bytes": h264.stat().st_size,
        }
        provenance_path = h264.with_suffix(h264.suffix + ".json")
        write_json(provenance_path, provenance)
        variants[key] = {
            "original": original,
            "h264": h264,
            "provenance": provenance_path,
        }

    dataset = {
        "run_id": run_id,
        "video_path": source,
        "tracking_dir": tracking_dir,
        "tracking_summary": tracking_summary,
        "detection_summary": detection_summary,
        "team_summary": team_summary,
        "compensation_summary": compensation_summary,
        "mapping_summary": mapping_summary,
        "source_summaries": {
            key: str(path) for key, (path, _) in summaries.items()
        },
    }
    return output, dataset, variants


def make_loader_fixture(root, run_id=RUN_ID, frame_range=(0, 100), source=None):
    """Build a fully linked tracking and derived-output dataset in a temp dir."""
    output, dataset, _ = make_video_fixture(root, run_id, frame_range, source)
    row = track_row(frame_range[0], run_id=run_id)
    write_csv(dataset["tracking_dir"] / "tracks_confirmed.csv", [row])
    write_csv(dataset["tracking_dir"] / "tracks.csv", [row])
    team_dir = output / f"teams_{run_id}"
    write_csv(team_dir / "team_assignments.csv", [{
        "run_id": run_id, "track_id": "1", "predicted_team": "team_a",
        "predicted_role": "team_a_outfield", "classification_confidence": 0.9,
        "number_of_samples": 1,
    }])
    comp_dir = output / f"compensation_{run_id}"
    comp_row = {
        **row, "camera_segment_id": "segment_000", "reference_frame": 0,
        "compensation_valid": True, "compensated_x": 10, "compensated_y": 20,
    }
    write_csv(comp_dir / "compensated_tracks.csv", [comp_row])
    map_dir = output / f"mapping_{run_id}"
    write_csv(map_dir / "mapped_player_positions.csv", [{
        **comp_row, "pitch_x_m": 1, "pitch_y_m": 2,
        "pitch_mapping_valid": True, "pitch_mapping_invalid_reason": "",
    }])
    detection_dir = output / f"detection_{run_id}"
    write_csv(detection_dir / "detections.csv", [{
        "frame_number": frame_range[0], "object_class": "person",
        "confidence": 0.8, "x1": 1, "y1": 2, "x2": 3, "y2": 4,
    }])
    return output, dataset


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

    def test_video_options_accept_only_run_specific_matching_h264_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
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
            self.assertEqual(options[2]["path"], variants["teams"]["h264"].resolve())

    def test_h264_selection_isolated_between_runs_with_same_source_and_different_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "same_match.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"same source footage")
            run_100 = "3a76cae39f9240f6a57a8000188e1eb1"
            run_368 = "9e5479f625d24b90bfc93042dceb13c8"
            output, dataset_100, variants_100 = make_video_fixture(
                root, run_100, (0, 100), source
            )
            _, dataset_368, variants_368 = make_video_fixture(
                root, run_368, (0, 368), source
            )

            selected_100 = dashboard.compatible_video_options(output, dataset_100)
            selected_368 = dashboard.compatible_video_options(output, dataset_368)
            tracking_100 = next(
                item for item in selected_100 if item["label"] == "ByteTrack tracking · H.264"
            )
            tracking_368 = next(
                item for item in selected_368 if item["label"] == "ByteTrack tracking · H.264"
            )
            self.assertEqual(tracking_100["path"], variants_100["tracking"]["h264"].resolve())
            self.assertEqual(tracking_368["path"], variants_368["tracking"]["h264"].resolve())
            self.assertNotEqual(tracking_100["path"], tracking_368["path"])

    def test_video_options_exclude_derived_artifacts_from_other_tracking_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            incompatible_summaries = {
                "teams": dataset["team_summary"],
                "compensation": dataset["compensation_summary"],
                "mapping": dataset["mapping_summary"],
            }
            for key, summary in incompatible_summaries.items():
                summary["tracking_run_id"] = "another-tracking-run"
                if key in {"teams", "compensation"}:
                    summary["tracking_csv"] = str(
                        Path(dataset["tracking_dir"]) / "tracks_from_another_run.csv"
                    )
                if key == "mapping":
                    summary["compensation_run_id"] = "another-compensation-run"

            options = dashboard.compatible_video_options(output, dataset)
            offered_paths = {option["path"] for option in options}
            offered_labels = {option["label"] for option in options}
            for key in ("teams", "compensation", "mapping"):
                self.assertNotIn(variants[key]["original"].resolve(), offered_paths)
                self.assertNotIn(variants[key]["h264"].resolve(), offered_paths)
                self.assertNotIn(
                    f"{ {'teams': 'Team classification', 'compensation': 'Camera-compensation comparison', 'mapping': 'Pitch mapping'}[key] } · original",
                    offered_labels,
                )
                self.assertNotIn(
                    f"{ {'teams': 'Team classification', 'compensation': 'Camera-compensation comparison', 'mapping': 'Pitch mapping'}[key] } · H.264",
                    offered_labels,
                )

            self.assertIn(variants["tracking"]["original"].resolve(), offered_paths)
            self.assertIn(dataset["video_path"].resolve(), offered_paths)
            self.assertIn("ByteTrack tracking · original", offered_labels)
            self.assertIn("Source video · match.mp4", offered_labels)

    def test_h264_with_mismatched_tracking_id_is_rejected_and_original_remains(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            provenance_path = variants["tracking"]["provenance"]
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["tracking_run_id"] = "another-run"
            write_json(provenance_path, provenance)

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_h264_with_mismatched_end_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            provenance_path = variants["tracking"]["provenance"]
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["end_frame_exclusive"] += 1
            write_json(provenance_path, provenance)

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_h264_with_mismatched_original_video_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            provenance_path = variants["tracking"]["provenance"]
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["original_video_sha256"] = "0" * 64
            write_json(provenance_path, provenance)

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_h264_with_mismatched_source_footage_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            provenance_path = variants["tracking"]["provenance"]
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["source_footage_sha256"] = "0" * 64
            write_json(provenance_path, provenance)

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_h264_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            variants["tracking"]["provenance"].unlink()

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_legacy_shared_h264_copy_is_not_assigned_to_a_tracking_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            variants["tracking"]["h264"].unlink()
            shared_copy = output / "browser_videos" / "annotated_tracking_h264.mp4"
            shared_copy.parent.mkdir(parents=True, exist_ok=True)
            shared_copy.write_bytes(b"legacy shared H264")

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_h264_with_mismatched_output_integrity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            variants["tracking"]["h264"].write_bytes(b"modified conversion output")

            options = dashboard.compatible_video_options(output, dataset)
            labels = [item["label"] for item in options]
            self.assertNotIn("ByteTrack tracking · H.264", labels)
            self.assertIn("ByteTrack tracking · original", labels)

    def test_missing_h264_file_preserves_original_and_source_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, variants = make_video_fixture(Path(temporary))
            variants["tracking"]["h264"].unlink()

            options = dashboard.compatible_video_options(output, dataset)
            tracking_original = next(
                item for item in options if item["label"] == "ByteTrack tracking · original"
            )
            source_video = next(
                item for item in options if item["label"].startswith("Source video ·")
            )
            self.assertEqual(tracking_original["path"], variants["tracking"]["original"].resolve())
            self.assertEqual(source_video["path"], dataset["video_path"].resolve())

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

    def test_derived_outputs_are_selected_by_provenance_not_folder_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "same_video.mp4"
            output, _ = make_loader_fixture(root, "run-100", (0, 100), source)
            make_loader_fixture(root, "run-368", (0, 368), source)
            for run_id, expected_end in (("run-100", 100), ("run-368", 368)):
                with self.subTest(run=run_id):
                    selected = dashboard.load_dashboard_dataset(
                        output, output / f"tracking_{run_id}"
                    )
                    self.assertEqual(selected["run_id"], run_id)
                    self.assertEqual(selected["mapping_summary"]["compensation_run_id"], f"comp-{run_id}")
                    self.assertEqual(selected["team_summary"]["tracking_run_id"], run_id)
                    self.assertEqual(selected["compensation_summary"]["tracking_run_id"], run_id)
                    self.assertEqual(selected["detection_summary"]["end_frame_exclusive"], expected_end)
                    self.assertEqual(selected["assignments"].iloc[0]["run_id"], run_id)
                    self.assertEqual(selected["mapping"].iloc[0]["run_id"], run_id)
                    self.assertEqual(selected["errors"], [])

    def test_ambiguous_derived_outputs_are_withheld_not_selected_by_name(self):
        cases = (
            ("teams", "team_classification_summary.json", "team_assignments.csv", "team_summary", "team-classification"),
            ("compensation", "compensation_summary.json", "compensated_tracks.csv", "compensation_summary", "camera-compensation"),
            ("mapping", "mapping_summary.json", "mapped_player_positions.csv", "mapping_summary", "pitch-mapping"),
            ("detection", "detection_summary.json", "detections.csv", "detection_summary", "detection"),
        )
        for subdir, summary_name, data_name, result_key, warning_name in cases:
            with self.subTest(analysis=subdir), tempfile.TemporaryDirectory() as temporary:
                output, _ = make_loader_fixture(Path(temporary))
                original = output / f"{subdir}_{RUN_ID}"
                duplicate = output / f"zz_duplicate_{subdir}"
                write_json(duplicate / summary_name, json.loads(
                    (original / summary_name).read_text(encoding="utf-8")
                ))
                (duplicate / data_name).write_bytes((original / data_name).read_bytes())
                selected = dashboard.load_dashboard_dataset(output, output / f"tracking_{RUN_ID}")
                self.assertIsNone(selected[result_key])
                self.assertTrue(any(
                    f"Multiple compatible {warning_name} outputs" in text
                    for text in selected["warnings"]
                ))
                if subdir == "teams":
                    self.assertIsNone(selected["assignments"])
                    self.assertEqual(set(selected["confirmed_tracks"]["predicted_team"]), {"unknown"})
                if subdir == "compensation":
                    self.assertIsNone(selected["mapping"])
                    self.assertTrue(any("mapping was withheld" in text for text in selected["warnings"]))
                if subdir == "mapping":
                    self.assertIsNone(selected["mapping"])
                if subdir == "detection":
                    self.assertIsNone(selected["detections"])
                self.assertIsNotNone(selected["confirmed_tracks"])
                self.assertEqual(selected["run_id"], RUN_ID)

    def test_mapping_is_not_selected_without_verified_compensation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = make_loader_fixture(Path(temporary))
            (output / f"compensation_{RUN_ID}" / "compensation_summary.json").unlink()
            selected = dashboard.load_dashboard_dataset(output, output / f"tracking_{RUN_ID}")
            self.assertIsNone(selected["compensation_summary"])
            self.assertIsNone(selected["mapping_summary"])
            self.assertIsNone(selected["mapping"])
            self.assertTrue(any("mapping was withheld" in text for text in selected["warnings"]))

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

    def test_cache_fingerprint_stays_stable_without_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, _ = make_video_fixture(Path(temporary))
            tracking_dir = dataset["tracking_dir"]
            before = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            self.assertEqual(before, dashboard.dataset_cache_fingerprint(output, tracking_dir))
            self.assertEqual(len(before), 64)

    def test_cache_fingerprint_changes_when_selected_tracking_csv_or_json_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, _ = make_video_fixture(Path(temporary))
            tracking_dir = dataset["tracking_dir"]
            csv_path = tracking_dir / "tracks_confirmed.csv"
            write_csv(csv_path, [track_row()])
            previous = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            write_csv(csv_path, [track_row(frame=1)])
            # Force an observable mtime change even on coarse timestamp filesystems.
            stat = csv_path.stat()
            os.utime(csv_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
            # A file edit is detectable even when its length is unchanged.
            self.assertNotEqual(previous, dashboard.dataset_cache_fingerprint(output, tracking_dir))

            previous = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            summary_path = tracking_dir / "tracking_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["frames_processed"] = 100
            write_json(summary_path, summary)
            self.assertNotEqual(previous, dashboard.dataset_cache_fingerprint(output, tracking_dir))

    def test_cache_fingerprint_changes_for_new_removed_and_edited_derived_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, dataset, _ = make_video_fixture(Path(temporary))
            tracking_dir = dataset["tracking_dir"]
            before = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            new_dir = output / "pitch_mapping" / "mapping_new"
            mapping_summary = new_dir / "mapping_summary.json"
            write_json(mapping_summary, {"tracking_run_id": RUN_ID})
            after_addition = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            self.assertNotEqual(before, after_addition)
            mapping_csv = new_dir / "mapped_player_positions.csv"
            mapping_csv.write_text("run_id,track_id\n", encoding="utf-8")
            after_csv = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            self.assertNotEqual(after_addition, after_csv)
            mapping_summary.unlink()
            self.assertNotEqual(after_csv, dashboard.dataset_cache_fingerprint(output, tracking_dir))

    def test_cache_fingerprint_covers_calibration_quality_and_run_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, dataset, _ = make_video_fixture(root)
            tracking_dir = dataset["tracking_dir"]
            before = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            pitch_calibration = output / "pitch_mapping" / "calibration" / "pitch_calibration.json"
            write_json(pitch_calibration, {"camera_segment_id": "segment_000"})
            self.assertNotEqual(before, dashboard.dataset_cache_fingerprint(output, tracking_dir))
            before = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            (output / "camera_quality_hybrid.csv").write_text("frame,valid\n0,True\n", encoding="utf-8")
            self.assertNotEqual(before, dashboard.dataset_cache_fingerprint(output, tracking_dir))
            # Merely adding an unrelated video does not trigger unnecessary reloads.
            before = dashboard.dataset_cache_fingerprint(output, tracking_dir)
            (output / "unrelated.mp4").write_bytes(b"video")
            self.assertEqual(before, dashboard.dataset_cache_fingerprint(output, tracking_dir))
            _, other_dataset, _ = make_video_fixture(root, run_id="another-run", frame_range=(0, 368))
            self.assertNotEqual(
                dashboard.dataset_cache_fingerprint(output, tracking_dir),
                dashboard.dataset_cache_fingerprint(output, other_dataset["tracking_dir"]),
            )

    def test_app_dataset_cache_is_keyed_by_file_fingerprint(self):
        # The Streamlit app itself cannot be imported here: it executes UI code
        # at module import time. Verify the fingerprint is passed as a cached
        # function argument, not ignored/underscored (which Streamlit excludes).
        app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(app_source)
        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_dataset"
        ]
        self.assertEqual(len(functions), 1)
        self.assertIn("file_fingerprint", [arg.arg for arg in functions[0].args.args])
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "load_dataset"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 3)
        self.assertIsInstance(calls[0].args[2], ast.Name)
        self.assertEqual(calls[0].args[2].id, "file_fingerprint")

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
