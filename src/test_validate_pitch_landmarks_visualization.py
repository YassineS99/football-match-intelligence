import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_pitch_landmarks as validator


class LandmarkVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.text_calls = []
        self.circle_calls = []
        self.line_calls = []

    def capture_text(
        self, image, text, origin, font, scale, color, thickness, line_type=None
    ):
        self.text_calls.append((text, origin, color))
        return image

    def capture_circle(self, image, center, radius, color, thickness):
        self.circle_calls.append((center, color))
        return image

    def capture_line(self, image, start, end, color, thickness):
        self.line_calls.append((start, end, color))
        return image

    def drawing_patches(self):
        return (
            patch.object(validator.cv2, "putText", side_effect=self.capture_text),
            patch.object(validator.cv2, "circle", side_effect=self.capture_circle),
            patch.object(validator.cv2, "line", side_effect=self.capture_line),
        )

    def test_alignment_labels_errors_and_connectors_follow_landmark_ids(self):
        reference = np.zeros((120, 240, 3), dtype=np.uint8)
        reference_points = np.array([[20.0, 30.0], [80.0, 70.0]])
        predicted = np.array([[24.0, 33.0], [100.0, 60.0]])
        errors = np.array([5.0, np.sqrt(500.0)])
        patches = self.drawing_patches()
        with patches[0], patches[1], patches[2]:
            result = validator._draw_alignment_image(
                reference, 50, reference_points, predicted, errors
            )

        self.assertEqual(result.shape, (194, 240, 3))
        for text, origin, _ in self.text_calls:
            if text in {"R1", "R2", "P1", "P2"}:
                self.assertEqual(
                    origin,
                    {
                        "R1": (30, 20),
                        "R2": (90, 60),
                        "P1": (34, 53),
                        "P2": (110, 80),
                    }[text],
                )
        labels = [text for text, _, _ in self.text_calls]
        for label in ("R1", "R2", "P1", "P2", "5.0 px", "22.4 px"):
            self.assertIn(label, labels)
        self.assertIn(((20, 30), (24, 33), (0, 255, 255)), self.line_calls)
        self.assertIn(((80, 70), (100, 60), (0, 255, 255)), self.line_calls)
        self.assertIn(((20, 30), (0, 255, 0)), self.circle_calls)
        self.assertIn(((24, 33), (0, 0, 255)), self.circle_calls)
        self.assertIn(((80, 70), (0, 255, 0)), self.circle_calls)
        self.assertIn(((100, 60), (0, 0, 255)), self.circle_calls)
        self.assertIn("R = reference", labels)
        self.assertIn("P = transformed prediction", labels)
        self.assertIn("matching pair", labels)

    def test_selected_points_image_labels_each_frame_with_matching_id(self):
        reference = np.zeros((100, 200, 3), dtype=np.uint8)
        current = np.zeros_like(reference)
        reference_points = np.array([[15.0, 25.0], [45.0, 55.0]])
        current_points = np.array([[115.0, 35.0], [145.0, 65.0]])
        patches = self.drawing_patches()
        with patches[0], patches[1]:
            result = validator._draw_selected_points_image(
                reference,
                current,
                50,
                reference_points,
                current_points,
            )

        self.assertEqual(result.shape, (154, 400, 3))
        label_origins = {
            text: origin
            for text, origin, _ in self.text_calls
            if text in {"R1", "R2", "T1", "T2"}
        }
        self.assertEqual(
            label_origins,
            {
                "R1": (25, 15),
                "R2": (55, 45),
                "T1": (125, 25),
                "T2": (155, 55),
            },
        )
        self.assertIn(((15, 25), (0, 255, 0)), self.circle_calls)
        self.assertIn(((45, 55), (0, 255, 0)), self.circle_calls)
        self.assertIn(((115, 35), (0, 255, 0)), self.circle_calls)
        self.assertIn(((145, 65), (0, 255, 0)), self.circle_calls)
        self.assertIn(
            "R and T labels share the same landmark ID",
            [text for text, _, _ in self.text_calls],
        )

    def test_selected_points_image_rejects_mismatched_frame_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            validator._draw_selected_points_image(
                np.zeros((100, 200, 3), dtype=np.uint8),
                np.zeros((90, 200, 3), dtype=np.uint8),
                50,
                np.zeros((1, 2)),
                np.zeros((1, 2)),
            )

    def test_save_run_writes_visualizations_to_unique_run_directories(self):
        reference = np.zeros((100, 200, 3), dtype=np.uint8)
        current = np.full_like(reference, 40)
        reference_points = np.array(
            [[15.0, 25.0], [45.0, 55.0], [75.0, 65.0], [105.0, 75.0]]
        )
        current_points = reference_points + np.array([2.0, 3.0])
        predicted = reference_points + np.array([1.0, 1.0])
        errors = np.linalg.norm(predicted - reference_points, axis=1)

        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory)
            with (
                patch.object(validator, "OUTPUT", output),
                patch.object(validator, "save_summary_chart"),
            ):
                first_run, _ = validator.save_run(
                    50, reference, current, reference_points, current_points,
                    predicted, errors
                )
                second_run, _ = validator.save_run(
                    50, reference, current, reference_points, current_points,
                    predicted, errors
                )

            self.assertNotEqual(first_run, second_run)
            for run_directory in (first_run, second_run):
                alignment = cv2.imread(
                    str(run_directory / "landmark_alignment.jpg")
                )
                selected = cv2.imread(
                    str(run_directory / "landmark_selected_points.jpg")
                )
                self.assertEqual(alignment.shape, (174, 200, 3))
                self.assertEqual(selected.shape, (154, 400, 3))

            with (first_run / "landmark_errors.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                error_rows = list(csv.DictReader(file))
            with (first_run / "selected_points.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                selected_rows = list(csv.DictReader(file))
            self.assertEqual(
                list(error_rows[0]),
                [
                    "reference_frame", "test_frame", "landmark_id",
                    "reference_x", "reference_y", "current_x", "current_y",
                    "predicted_reference_x", "predicted_reference_y", "error_px",
                ],
            )
            self.assertEqual(
                list(selected_rows[0]),
                [
                    "reference_frame", "test_frame", "landmark_id",
                    "reference_x", "reference_y", "current_x", "current_y",
                ],
            )
            self.assertEqual(len(error_rows), 4)
            self.assertEqual(len(selected_rows), 4)


if __name__ == "__main__":
    unittest.main()
