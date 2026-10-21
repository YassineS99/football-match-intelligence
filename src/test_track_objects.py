import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_objects as tracking


def person_detection(x=100.0, confidence=0.9):
    return {
        "class_id": tracking.PERSON_CLASS_ID,
        "object_class": "person",
        "confidence": confidence,
        "x1": x,
        "y1": 100.0,
        "x2": x + 40.0,
        "y2": 200.0,
    }


def ball_detection():
    return {
        "class_id": tracking.SPORTS_BALL_CLASS_ID,
        "object_class": "sports ball",
        "confidence": 0.8,
        "x1": 10.0,
        "y1": 20.0,
        "x2": 20.0,
        "y2": 30.0,
    }


class TrackObjectsTests(unittest.TestCase):
    def test_tracking_csv_schema(self):
        self.assertEqual(
            tracking.TRACK_CSV_FIELDS,
            [
                "run_id", "frame", "timestamp", "track_id", "class_id", "confidence",
                "x1", "y1", "x2", "y2", "center_x", "center_y", "foot_x",
                "foot_y",
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracks.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=tracking.TRACK_CSV_FIELDS).writeheader()
            with path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(
                    csv.DictReader(file).fieldnames, tracking.TRACK_CSV_FIELDS
                )

    def test_cli_argument_validation(self):
        parser = tracking.build_parser()
        invalid_confidence = parser.parse_args(["--conf", "0"])
        with self.assertRaisesRegex(ValueError, "--detection-conf"):
            tracking.validate_args(invalid_confidence)
        invalid_start = parser.parse_args(["--start-frame", "-1"])
        with self.assertRaisesRegex(ValueError, "--start-frame"):
            tracking.validate_args(invalid_start)
        invalid_limit = parser.parse_args(["--max-frames", "0"])
        with self.assertRaisesRegex(ValueError, "--max-frames"):
            tracking.validate_args(invalid_limit)
        invalid_display_confidence = parser.parse_args(["--display-conf", "1.1"])
        with self.assertRaisesRegex(ValueError, "--display-conf"):
            tracking.validate_args(invalid_display_confidence)
        invalid_duration = parser.parse_args(["--min-confirmed-frames", "0"])
        with self.assertRaisesRegex(ValueError, "--min-confirmed-frames"):
            tracking.validate_args(invalid_duration)

    def test_run_ids_are_unique_and_explicitly_scoped(self):
        first = tracking.create_tracking_run_id()
        second = tracking.create_tracking_run_id()
        self.assertNotEqual(first, second)
        self.assertIn("run_id", tracking.TRACK_CSV_FIELDS)

    def test_detection_and_association_thresholds_are_separate(self):
        config = tracking.load_tracker_config()
        self.assertEqual(
            tracking.DEFAULT_MIN_CONFIRMED_FRAMES, 10
        )
        self.assertEqual(
            tracking.resolve_detection_confidence(None, config),
            config.track_low_thresh,
        )
        self.assertEqual(config.track_low_thresh, 0.1)
        with self.assertRaisesRegex(ValueError, "lower-confidence boxes are missing"):
            tracking.validate_precomputed_confidence(0.35, 0.1)
        tracking.validate_precomputed_confidence(0.35, 0.35)
        self.assertEqual(
            tracking.resolve_detection_confidence(0.35, config), 0.35
        )
        overridden = tracking.load_tracker_config({
            "track_low_thresh": 0.08,
            "track_high_thresh": 0.3,
            "new_track_thresh": 0.28,
        })
        self.assertEqual(
            tracking.resolve_detection_confidence(None, overridden), 0.08
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            tracking.load_tracker_config({
                "track_low_thresh": 0.4,
                "track_high_thresh": 0.3,
            })

    def test_tracking_ids_persist_across_consecutive_mocked_detections(self):
        session = tracking.TrackingSession.create()
        frame_shape = (720, 1280, 3)
        first = tracking.update_tracking_frame(
            session, [person_detection()], 10, 25.0, frame_shape
        )
        second = tracking.update_tracking_frame(
            session, [person_detection(x=101.0)], 11, 25.0, frame_shape
        )
        third = tracking.update_tracking_frame(
            session, [person_detection(x=102.0)], 12, 25.0, frame_shape
        )
        self.assertNotEqual(first[0]["track_id"], "")
        self.assertEqual(second[0]["track_id"], third[0]["track_id"])
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertEqual(second[0]["frame"], 11)
        self.assertEqual(third[0]["frame"], 12)

    def test_empty_frames_advance_tracker_without_emitting_rows(self):
        session = tracking.TrackingSession.create()
        frame_shape = (720, 1280, 3)
        rows = tracking.update_tracking_frame(
            session, [], 3, 25.0, frame_shape
        )
        self.assertEqual(rows, [])
        self.assertEqual(session.last_frame, 3)
        tracked = tracking.update_tracking_frame(
            session, [person_detection()], 4, 25.0, frame_shape
        )
        self.assertEqual(len(tracked), 1)

    def test_original_coordinates_timestamp_and_foot_point(self):
        session = tracking.TrackingSession.create()
        rows = tracking.update_tracking_frame(
            session, [person_detection(x=125.5)], 50, 25.0, (720, 1280, 3)
        )
        row = rows[0]
        self.assertEqual(row["frame"], 50)
        self.assertEqual(row["timestamp"], 2.0)
        self.assertEqual((row["x1"], row["y1"], row["x2"], row["y2"]),
                         (125.5, 100.0, 165.5, 200.0))
        self.assertEqual((row["center_x"], row["center_y"]), (145.5, 150.0))
        self.assertEqual((row["foot_x"], row["foot_y"]), (145.5, 200.0))

    def test_discontinuous_frame_starts_new_noncolliding_id_segment(self):
        session = tracking.TrackingSession.create()
        shape = (720, 1280, 3)
        tracking.update_tracking_frame(session, [person_detection()], 20, 25.0, shape)
        old_id = tracking.update_tracking_frame(
            session, [person_detection(x=101.0)], 21, 25.0, shape
        )[0]["track_id"]
        tracking.update_tracking_frame(session, [], 23, 25.0, shape)
        tracking.update_tracking_frame(
            session, [person_detection()], 24, 25.0, shape
        )
        new_id = tracking.update_tracking_frame(
            session, [person_detection(x=101.0)], 25, 25.0, shape
        )[0]["track_id"]
        self.assertNotEqual(old_id, "")
        self.assertNotEqual(new_id, "")
        self.assertNotEqual(old_id, new_id)

    def test_missing_ids_are_saved_and_counted(self):
        session = tracking.TrackingSession.create()
        rows = tracking.update_tracking_frame(
            session,
            [person_detection(confidence=0.15), ball_detection()],
            0,
            25.0,
            (720, 1280, 3),
        )
        self.assertEqual(rows[0]["track_id"], "")
        self.assertEqual(rows[1]["track_id"], "")
        stats = tracking.TrackStatistics()
        for row in rows:
            stats.add(row)
        summary = stats.summary()
        self.assertEqual(summary["person_detections_without_track_id"], 1)
        self.assertEqual(summary["sports_ball_detections_untracked"], 1)
        self.assertEqual(summary["person_detections_with_track_id"], 0)
        self.assertEqual(summary["total_tracked_detections"], 0)

    def test_duration_and_fragmentation_statistics(self):
        stats = tracking.TrackStatistics()
        for frame in [0, 1, 2, 4, 5]:
            stats.add({
                "frame": frame,
                "class_id": tracking.PERSON_CLASS_ID,
                "track_id": 7,
            })
        summary = stats.summary()
        self.assertEqual(summary["unique_track_ids"], 1)
        self.assertEqual(summary["tracks_lasting_at_least_5_frames"], 1)
        self.assertEqual(summary["tracks_lasting_at_least_10_frames"], 0)
        self.assertEqual(summary["tracks_with_multiple_fragments"], 1)
        self.assertEqual(summary["total_extra_track_fragments"], 1)

    def test_short_tracks_are_hidden_from_confirmed_view_but_preserved_raw(self):
        stats = tracking.TrackStatistics()
        raw_rows = []
        for frame in range(35, 40):
            row = {
                "frame": frame,
                "class_id": tracking.PERSON_CLASS_ID,
                "track_id": 41,
                "confidence": 0.49,
                "x1": 280.0,
                "y1": 670.0,
                "x2": 305.0,
                "y2": 705.0,
            }
            raw_rows.append(row.copy())
            stats.add(row)
        for frame in range(10, 26):
            row = {
                "frame": frame,
                "class_id": tracking.PERSON_CLASS_ID,
                "track_id": 7,
                "confidence": 0.8,
                "x1": 500.0,
                "y1": 300.0,
                "x2": 530.0,
                "y2": 380.0,
            }
            raw_rows.append(row)
            stats.add(row)

        confirmed = stats.confirmed_track_ids(
            tracking.DEFAULT_MIN_CONFIRMED_FRAMES
        )
        visible = tracking.select_visualization_rows(
            raw_rows, confirmed, confirmed_only=True, display_confidence=0.0
        )
        self.assertNotIn(41, {row["track_id"] for row in visible})
        self.assertIn(7, {row["track_id"] for row in visible})
        self.assertEqual(sum(row["track_id"] == 41 for row in raw_rows), 5)
        self.assertEqual(len(raw_rows), 21)

    def test_short_unconfirmed_player_is_still_available_when_not_confirmed_only(self):
        row = {
            "frame": 8,
            "class_id": tracking.PERSON_CLASS_ID,
            "track_id": 3,
            "confidence": 0.6,
        }
        rows = [row]
        visible = tracking.select_visualization_rows(
            rows, set(), confirmed_only=False, display_confidence=0.0
        )
        self.assertEqual(visible, rows)
        self.assertEqual(rows, [row])

    def test_small_person_box_is_not_filtered_by_geometry(self):
        small_person = person_detection(x=10.0)
        small_person.update({"x2": 22.0, "y2": 118.0})
        session = tracking.TrackingSession.create()
        stats = tracking.TrackStatistics()
        rows = []
        for frame in range(tracking.DEFAULT_MIN_CONFIRMED_FRAMES):
            frame_rows = tracking.update_tracking_frame(
                session,
                [small_person],
                frame,
                25.0,
                (720, 1280, 3),
            )
            rows.extend(frame_rows)
            for row in frame_rows:
                stats.add(row)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["x2"] - rows[0]["x1"], 12.0)
        self.assertEqual(rows[0]["y2"] - rows[0]["y1"], 18.0)
        self.assertNotEqual(rows[0]["track_id"], "")
        confirmed = stats.confirmed_track_ids(
            tracking.DEFAULT_MIN_CONFIRMED_FRAMES
        )
        self.assertIn(int(rows[0]["track_id"]), confirmed)
        visible = tracking.select_visualization_rows(
            rows, confirmed, True, 0.0
        )
        self.assertEqual(visible, rows)

    def test_visualization_confidence_does_not_mutate_raw_rows_and_ball_remains_distinct(self):
        person = {
            "frame": 0,
            "class_id": tracking.PERSON_CLASS_ID,
            "track_id": 9,
            "confidence": 0.3,
        }
        ball = {
            "frame": 0,
            "class_id": tracking.SPORTS_BALL_CLASS_ID,
            "track_id": "",
            "confidence": 0.8,
        }
        raw_rows = [person.copy(), ball.copy()]
        before = [row.copy() for row in raw_rows]
        visible = tracking.select_visualization_rows(
            raw_rows, {9}, True, display_confidence=0.5
        )
        self.assertEqual(visible, [ball])
        self.assertEqual(raw_rows, before)

    def test_confirmed_csv_is_additional_and_does_not_remove_raw_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.csv"
            confirmed_path = Path(temp) / "confirmed.csv"
            raw_rows = [
                {
                    "run_id": "run-a",
                    "frame": 1,
                    "timestamp": 0.04,
                    "track_id": "2",
                    "class_id": 0,
                    "confidence": 0.8,
                    "x1": 10,
                    "y1": 20,
                    "x2": 30,
                    "y2": 60,
                    "center_x": 20,
                    "center_y": 40,
                    "foot_x": 20,
                    "foot_y": 60,
                },
                {
                    "run_id": "run-a",
                    "frame": 2,
                    "timestamp": 0.08,
                    "track_id": "3",
                    "class_id": 0,
                    "confidence": 0.4,
                    "x1": 40,
                    "y1": 50,
                    "x2": 50,
                    "y2": 70,
                    "center_x": 45,
                    "center_y": 60,
                    "foot_x": 45,
                    "foot_y": 70,
                },
            ]
            with raw_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=tracking.TRACK_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(raw_rows)
            count = tracking.write_confirmed_tracks(
                raw_path, confirmed_path, {2}
            )
            self.assertEqual(count, 1)
            with raw_path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            with confirmed_path.open(newline="", encoding="utf-8") as file:
                confirmed_rows = list(csv.DictReader(file))
            self.assertEqual(len(confirmed_rows), 1)
            self.assertEqual(confirmed_rows[0]["track_id"], "2")

    def test_balls_are_not_sent_to_person_tracker_and_rendered_separately(self):
        session = tracking.TrackingSession.create()
        rows = tracking.update_tracking_frame(
            session,
            [ball_detection()],
            4,
            25.0,
            (720, 1280, 3),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_id"], tracking.SPORTS_BALL_CLASS_ID)
        self.assertEqual(rows[0]["track_id"], "")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        annotated = tracking.draw_tracking(frame, rows, {})
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))

    def test_video_output_dimensions_and_mp4_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (128, 72),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((72, 128, 3), dtype=np.uint8))
            writer.release()
            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertEqual(
                (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
                (128, 72),
            )
            ok, _ = capture.read()
            capture.release()
            self.assertTrue(ok)

    def test_unique_output_path_creation_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            final_paths, temporary_paths = tracking.create_temp_paths(output_dir)
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
            self.assertEqual(
                set(final_paths),
                set(tracking.OUTPUT_FILENAMES),
            )
            (output_dir / "tracks.csv").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                tracking.create_temp_paths(output_dir)

    def test_detection_csv_metadata_and_frame_iteration(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            detections = directory / "detections.csv"
            metadata = directory / "detection_summary.json"
            metadata.write_text(json.dumps({
                "input_video": str(Path("sample.mp4").resolve()),
                "start_frame": 0,
                "end_frame_exclusive": 3,
                "model_classes": {"0": "person", "32": "sports ball"},
            }), encoding="utf-8")
            with detections.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "frame_number", "object_class", "confidence",
                        "x1", "y1", "x2", "y2",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "frame_number": 0, "object_class": "person",
                    "confidence": 0.8, "x1": 1, "y1": 2, "x2": 10, "y2": 20,
                })
                writer.writerow({
                    "frame_number": 2, "object_class": "sports ball",
                    "confidence": 0.7, "x1": 4, "y1": 5, "x2": 8, "y2": 9,
                })
            input_path = Path("sample.mp4").resolve()
            _, start, end, names = tracking._read_detection_metadata(
                detections, input_path
            )
            self.assertEqual(tracking.validate_detection_csv(detections), 2)
            frames = list(tracking.iter_detection_frames(
                detections, start, end, names, 0.35
            ))
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(frames[0]), 1)
            self.assertEqual(frames[1], [])
            self.assertEqual(frames[2][0]["class_id"], 32)


if __name__ == "__main__":
    unittest.main()
