"""Stage 3 frame explorer: isolated runs, read-only video, finite geometry."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import dashboard_data as dashboard
import frame_explorer as explorer


class FrameExplorerTests(unittest.TestCase):
    def tracks(self, run="run-a"):
        return pd.DataFrame([
            {"run_id": run, "frame": 0, "class_id": 0, "track_id": "1",
             "predicted_role": "team_a_outfield", "foot_x": 15, "foot_y": 30,
             "x1": 8, "y1": 9, "x2": 25, "y2": 32},
            {"run_id": run, "frame": 0, "class_id": 0, "track_id": "2",
             "predicted_role": "unknown", "foot_x": 75, "foot_y": 40,
             "x1": 65, "y1": 19, "x2": 87, "y2": 42},
            {"run_id": run, "frame": 1, "class_id": 0, "track_id": "1",
             "predicted_role": "team_a_outfield", "foot_x": 17, "foot_y": 33},
            {"run_id": run, "frame": 0, "class_id": 32, "track_id": "99",
             "predicted_role": "unknown", "foot_x": 7, "foot_y": 10},
        ])

    def test_selected_frame_and_person_class_only(self):
        result = dashboard.frame_explorer_rows(self.tracks(), "run-a", 0)
        self.assertEqual(result["track_id"].tolist(), ["1", "2"])
        self.assertTrue(result["frame"].eq(0).all())
        self.assertTrue(result["class_id"].eq(0).all())

    def test_frame_and_provisional_role_and_track_filters(self):
        data = self.tracks()
        first = dashboard.frame_explorer_rows(data, "run-a", 0, roles=["team_a_outfield"])
        self.assertEqual(first["track_id"].tolist(), ["1"])
        self.assertEqual(len(dashboard.frame_explorer_rows(data, "run-a", 0, roles=[])), 0)
        self.assertEqual(len(dashboard.frame_explorer_rows(data, "run-a", 0, track_id="2")), 1)
        self.assertEqual(len(dashboard.frame_explorer_rows(data, "run-a", 10)), 0)

    def test_identical_track_id_other_run_is_rejected(self):
        tracks = pd.concat([self.tracks("run-a"), self.tracks("run-b")], ignore_index=True)
        with self.assertRaisesRegex(dashboard.DataValidationError, "multiple tracking run"):
            dashboard.frame_explorer_rows(tracks, "run-a", 0)
        with self.assertRaisesRegex(dashboard.DataValidationError, "does not match"):
            dashboard.frame_explorer_rows(self.tracks("run-b"), "run-a", 0)

    def test_negative_frames_are_rejected(self):
        with self.assertRaises(dashboard.DataValidationError):
            dashboard.frame_explorer_rows(self.tracks(), "run-a", -1)

    def test_frame_annotation_changes_only_copy(self):
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        before = frame.copy()
        result = explorer.draw_frame_annotations(frame, self.tracks().iloc[:2])
        np.testing.assert_array_equal(frame, before)
        self.assertTrue(np.any(result != before))
        self.assertEqual(result.shape, frame.shape)

    def test_missing_boxes_falls_back_to_footpoints(self):
        rows = self.tracks().iloc[:1].drop(columns=["x1", "y1", "x2", "y2"])
        blank = np.zeros((60, 100, 3), dtype=np.uint8)
        result = explorer.draw_frame_annotations(blank, rows)
        self.assertTrue(np.any(result != blank))
        np.testing.assert_array_equal(blank, explorer.draw_frame_annotations(blank, rows, boxes=False, foot_points=False))

    def test_nonfinite_and_out_of_view_positions_are_not_drawn(self):
        rows = self.tracks().iloc[:1].copy()
        for column in ("x1", "y1", "x2", "y2", "foot_x", "foot_y"):
            rows[column] = np.nan
        blank = np.zeros((60, 100, 3), dtype=np.uint8)
        np.testing.assert_array_equal(blank, explorer.draw_frame_annotations(blank, rows))

    def test_read_source_frame_uses_requested_index_and_leaves_video_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frames.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            if not writer.isOpened():
                self.skipTest("MJPG encoder unavailable on this OpenCV build")
            try:
                for level in (20, 110, 230):
                    writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
            finally:
                writer.release()
            before = path.read_bytes()
            first = explorer.read_source_frame(path, 0)
            last = explorer.read_source_frame(path, 2)
            self.assertEqual(first.shape, (48, 64, 3))
            self.assertGreater(float(last.mean()), float(first.mean()) + 150)
            self.assertEqual(before, path.read_bytes())

    def test_missing_source_and_out_of_range_frame_are_handled(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                explorer.read_source_frame(Path(temporary) / "missing.mp4", 0)
        with self.assertRaises(ValueError):
            explorer.read_source_frame("not-needed.mp4", -1)


if __name__ == "__main__":
    unittest.main()
