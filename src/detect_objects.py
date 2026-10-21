import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "input" / "sample.mp4"
DEFAULT_MODEL = ROOT / "yolov8x.pt"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "detection"
DEFAULT_CONFIDENCE = 0.35
RANDOM_SEED = 42
PERSON_CLASS = "person"
BALL_CLASS = "sports ball"
TARGET_CLASSES = {PERSON_CLASS, BALL_CLASS}

CSV_FIELDS = [
    "frame_number",
    "timestamp_seconds",
    "object_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "bottom_center_x",
    "bottom_center_y",
]
OUTPUT_FILENAMES = (
    "annotated_detection.mp4",
    "detections.csv",
    "detection_summary.json",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Detect COCO person and sports-ball objects in a video. "
            "Person detections are provisional and do not distinguish "
            "players, referees, or goalkeepers."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input video (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Local Ultralytics YOLO weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Detection confidence in (0, 1] (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, 0, or another CUDA device index (default: auto).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based first video frame to process (default: 0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all remaining frames).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def validate_args(args, parser=None):
    if not math.isfinite(args.conf) or not 0.0 < args.conf <= 1.0:
        message = "--conf must be greater than 0 and at most 1."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if args.start_frame < 0:
        message = "--start-frame must be a non-negative integer."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if args.max_frames is not None and args.max_frames <= 0:
        message = "--max-frames must be a positive integer when specified."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not str(args.device).strip():
        message = "--device cannot be empty."
        if parser:
            parser.error(message)
        raise ValueError(message)
    if not args.output_dir:
        message = "--output-dir cannot be empty."
        if parser:
            parser.error(message)
        raise ValueError(message)


def resolve_device(device):
    import torch

    requested = str(device).strip().lower()
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    try:
        index = int(requested)
    except ValueError as exc:
        raise ValueError(
            "--device must be auto, cpu, or a CUDA device index such as 0."
        ) from exc
    if index < 0:
        raise ValueError("CUDA device index must be non-negative.")
    if not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device {index} was requested, but CUDA is not available."
        )
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {index} is unavailable; "
            f"{torch.cuda.device_count()} device(s) detected."
        )
    return str(index)


def seed_generators(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_class_names(names):
    if isinstance(names, dict):
        return {int(class_id): str(name).strip().lower() for class_id, name in names.items()}
    return {index: str(name).strip().lower() for index, name in enumerate(names)}


def target_class_ids(names):
    normalized = normalize_class_names(names)
    selected = {
        class_id: class_name
        for class_id, class_name in normalized.items()
        if class_name in TARGET_CLASSES
    }
    missing = TARGET_CLASSES - set(selected.values())
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"Model does not provide required COCO classes: {missing_names}."
        )
    return selected


def _to_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def make_detection_record(
    frame_number, fps, object_class, confidence, box
):
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    is_person = object_class == PERSON_CLASS
    return {
        "frame_number": int(frame_number),
        "timestamp_seconds": float(frame_number / fps),
        "object_class": object_class,
        "confidence": float(confidence),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": center_x,
        "center_y": center_y,
        "bottom_center_x": center_x if is_person else "",
        "bottom_center_y": y2 if is_person else "",
    }


def records_from_result(result, class_names, frame_number, fps):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    class_ids = _to_numpy(boxes.cls).reshape(-1).astype(int)
    if not (len(xyxy) == len(confidences) == len(class_ids)):
        raise ValueError("Model returned mismatched box, confidence, and class arrays.")

    records = []
    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        object_class = class_names.get(int(class_id))
        if object_class not in TARGET_CLASSES:
            continue
        records.append(
            make_detection_record(
                frame_number, fps, object_class, confidence, box
            )
        )
    return records


def draw_detections(frame, records):
    annotated = frame.copy()
    height, width = frame.shape[:2]
    for record in records:
        x1 = int(np.clip(round(record["x1"]), 0, width - 1))
        y1 = int(np.clip(round(record["y1"]), 0, height - 1))
        x2 = int(np.clip(round(record["x2"]), 0, width - 1))
        y2 = int(np.clip(round(record["y2"]), 0, height - 1))
        color = (0, 220, 0) if record["object_class"] == PERSON_CLASS else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{record['object_class']} {record['confidence']:.2f}"
        label_y = max(18, y1 - 7)
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def prepare_output_paths(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing detection outputs: "
            + ", ".join(existing)
            + ". Choose a new --output-dir or move the previous run."
        )
    temporary = {
        "video": output_dir / f".annotated_detection.{random.randrange(1 << 32):08x}.tmp.mp4",
        "csv": output_dir / f".detections.{random.randrange(1 << 32):08x}.tmp.csv",
        "summary": output_dir / f".detection_summary.{random.randrange(1 << 32):08x}.tmp.json",
    }
    return paths, temporary


def detect_video(args):
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Local YOLO model weights not found: {model_path}. "
            "Provide an existing local weights file with --model."
        )

    paths, temporary = prepare_output_paths(output_dir)
    device = resolve_device(args.device)
    seed_generators()

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = target_class_ids(model.names)
    class_ids = sorted(names)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    writer = None
    processed = 0
    detections_total = 0
    class_counts = Counter()
    start_frame = args.start_frame
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid frame count or dimensions.")
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Input video has invalid FPS: {fps}.")
        if start_frame >= frame_count:
            raise ValueError(
                f"--start-frame {start_frame} is outside the video; "
                f"valid frames are 0 through {frame_count - 1}."
            )

        remaining = frame_count - start_frame
        frame_limit = min(args.max_frames or remaining, remaining)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(f"Could not seek to frame {start_frame}.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temporary["video"]), fourcc, fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not initialize MP4 writer at {temporary['video']}."
            )

        with temporary["csv"].open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            progress = tqdm(
                total=frame_limit,
                desc="Detecting",
                unit="frame",
                file=sys.stderr,
            )
            try:
                for frame_offset in range(frame_limit):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_number = start_frame + frame_offset
                    result = model.predict(
                        source=frame,
                        classes=class_ids,
                        conf=args.conf,
                        device=device,
                        verbose=False,
                    )[0]
                    records = records_from_result(
                        result, names, frame_number, fps
                    )
                    for record in records:
                        csv_writer.writerow(record)
                        class_counts[record["object_class"]] += 1
                    detections_total += len(records)
                    annotated = draw_detections(frame, records)
                    writer.write(annotated)
                    processed += 1
                    progress.update(1)
            finally:
                progress.close()

        if processed == 0:
            raise RuntimeError(
                f"No frames could be read starting at frame {start_frame}."
            )
        writer.release()
        writer = None
        capture.release()

        summary = {
            "input_video": str(input_path),
            "model": str(model_path),
            "model_classes": names,
            "class_limitation": (
                "COCO person detections are provisional and do not distinguish "
                "football players, referees, or goalkeepers."
            ),
            "confidence_threshold": args.conf,
            "device": device,
            "video_fps": fps,
            "frame_width": width,
            "frame_height": height,
            "source_frame_count": frame_count,
            "start_frame": start_frame,
            "frames_processed": processed,
            "end_frame_exclusive": start_frame + processed,
            "detections_total": detections_total,
            "detections_by_class": dict(class_counts),
            "csv_columns": CSV_FIELDS,
            "timestamp_basis": "frame_number / source_video_fps",
            "random_seed": RANDOM_SEED,
        }
        with temporary["summary"].open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")

        for key, final_path in (
            ("video", paths["annotated_detection.mp4"]),
            ("csv", paths["detections.csv"]),
            ("summary", paths["detection_summary.json"]),
        ):
            if final_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite detection output: {final_path}"
                )
            temporary[key].replace(final_path)

        return summary
    except Exception:
        if writer is not None:
            writer.release()
        capture.release()
        for temporary_path in temporary.values():
            if temporary_path.exists():
                temporary_path.unlink()
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args, parser)
        summary = detect_video(args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Detections: {summary['detections_total']}")
    print(f"Saved video: {args.output_dir / 'annotated_detection.mp4'}")
    print(f"Saved CSV: {args.output_dir / 'detections.csv'}")
    print(f"Saved summary: {args.output_dir / 'detection_summary.json'}")


if __name__ == "__main__":
    main()
