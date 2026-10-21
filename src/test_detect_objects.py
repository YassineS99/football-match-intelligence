import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_objects as detector


class DetectionPipelineTests(unittest.TestCase):
    def test_person_box_conversion_and_foot_point(self):
        record = detector.make_detection_record(
            frame_number=25,
            fps=25.0,
            object_class="person",
            confidence=0.875,
            box=[10.0, 20.0, 50.0, 100.0],
        )
        self.assertEqual(record["frame_number"], 25)
        self.assertEqual(record["timestamp_seconds"], 1.0)
        self.assertEqual(record["center_x"], 30.0)
        self.assertEqual(record["center_y"], 60.0)
        self.assertEqual(record["bottom_center_x"], 30.0)
        self.assertEqual(record["bottom_center_y"], 100.0)

    def test_ball_box_does_not_receive_person_foot_point(self):
        record = detector.make_detection_record(
            10, 20.0, "sports ball", 0.5, [2.0, 4.0, 8.0, 12.0]
        )
        self.assertEqual(record["center_x"], 5.0)
        self.assertEqual(record["center_y"], 8.0)
        self.assertEqual(record["bottom_center_x"], "")
        self.assertEqual(record["bottom_center_y"], "")

    def test_empty_result_is_handled_without_records(self):
        names = detector.target_class_ids(
            {0: "person", 32: "sports ball"}
        )
        self.assertEqual(
            detector.records_from_result(
                SimpleNamespace(boxes=None), names, 0, 25.0
            ),
            [],
        )
        class EmptyBoxes:
            xyxy = np.empty((0, 4))
            conf = np.empty((0,))
            cls = np.empty((0,))

            def __len__(self):
                return 0

        self.assertEqual(
            detector.records_from_result(
                SimpleNamespace(boxes=EmptyBoxes()), names, 0, 25.0
            ),
            [],
        )

    def test_records_filter_to_supported_classes_and_preserve_frame_time(self):
        class Boxes:
            xyxy = np.array([[1, 2, 11, 22], [3, 4, 9, 12], [0, 0, 5, 5]])
            conf = np.array([0.9, 0.7, 0.8])
            cls = np.array([0, 32, 1])

            def __len__(self):
                return len(self.xyxy)

        boxes = Boxes()
        result = SimpleNamespace(boxes=boxes)
        names = detector.target_class_ids(
            {0: "person", 1: "bicycle", 32: "sports ball"}
        )
        records = detector.records_from_result(result, names, 51, 30.0)
        self.assertEqual([row["object_class"] for row in records], ["person", "sports ball"])
        self.assertEqual([row["frame_number"] for row in records], [51, 51])
        self.assertEqual([row["timestamp_seconds"] for row in records], [1.7, 1.7])
        self.assertEqual(records[0]["bottom_center_x"], 6.0)
        self.assertEqual(records[0]["bottom_center_y"], 22.0)
        self.assertEqual(records[1]["bottom_center_x"], "")

    def test_csv_schema_and_empty_csv_header(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "detections.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=detector.CSV_FIELDS)
                writer.writeheader()
            with csv_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, detector.CSV_FIELDS)
                self.assertEqual(list(reader), [])

    def test_cli_argument_validation(self):
        parser = detector.build_parser()
        args = parser.parse_args([
            "--input", "clip.mp4",
            "--conf", "0.35",
            "--model", "weights.pt",
            "--device", "cpu",
            "--start-frame", "7",
            "--max-frames", "10",
            "--output-dir", "out",
        ])
        detector.validate_args(args)
        self.assertEqual(args.start_frame, 7)
        self.assertEqual(args.max_frames, 10)
        for option in (
            SimpleNamespace(**{**vars(args), "conf": 0.0}),
            SimpleNamespace(**{**vars(args), "conf": float("nan")}),
            SimpleNamespace(**{**vars(args), "start_frame": -1}),
            SimpleNamespace(**{**vars(args), "max_frames": 0}),
            SimpleNamespace(**{**vars(args), "device": ""}),
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                detector.validate_args(option)

    def test_model_must_include_person_and_sports_ball(self):
        self.assertEqual(
            detector.target_class_ids({0: "person", 32: "sports ball"}),
            {0: "person", 32: "sports ball"},
        )
        with self.assertRaisesRegex(ValueError, "sports ball"):
            detector.target_class_ids({0: "person", 1: "bicycle"})


if __name__ == "__main__":
    unittest.main()
