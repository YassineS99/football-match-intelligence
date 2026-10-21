import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


VIDEO = Path("data/input/sample.mp4")
TRANSFORMS = Path("data/output/camera_transforms_hybrid.npy")
QUALITY = Path("data/output/camera_quality_hybrid.csv")
OUTPUT = Path("data/output/landmark_validation")
REFERENCE_FRAME = 0
DEFAULT_TEST_FRAME = 50
DISPLAY_WIDTH = 800
MIN_LANDMARKS = 4
HEADER_HEIGHT = 140

SUMMARY_FIELDS = [
    "run_id",
    "created_at",
    "reference_frame",
    "test_frame",
    "landmark_count",
    "mean_error_px",
    "median_error_px",
    "max_error_px",
    "p90_error_px",
    "run_directory",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate hybrid camera alignment using pitch landmarks."
    )
    parser.add_argument(
        "--test-frame",
        type=int,
        default=DEFAULT_TEST_FRAME,
        help=(
            "Test frame number (default: 50; commonly used: "
            "25, 50, 100, 150, 200)."
        ),
    )
    return parser, parser.parse_args()


def load_transforms(path):
    if not path.is_file():
        raise FileNotFoundError(f"Camera transform file not found: {path}")

    transforms = np.load(path, allow_pickle=False)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected camera transforms with shape (frames, 3, 3), "
            f"got {transforms.shape}"
        )
    if not np.issubdtype(transforms.dtype, np.number) or np.iscomplexobj(transforms):
        raise ValueError("Camera transforms must contain real numeric values.")
    return transforms


def load_transition_quality(path, frame_count, transform_count):
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera quality file not found: {path}; it is required to "
            "detect failed camera transitions."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"frame", "method", "valid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: frame, method, valid"
            )

        rows = list(reader)

    if len(rows) != frame_count or len(rows) != transform_count:
        raise ValueError(
            "Video, camera transforms, and camera quality row counts differ "
            f"({frame_count}, {transform_count}, {len(rows)})."
        )

    failed_frames = []
    for expected_frame, row in enumerate(rows):
        try:
            frame_number = int(row["frame"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid frame number in camera quality row {expected_frame}"
            ) from exc

        if frame_number != expected_frame:
            raise ValueError(
                "Camera quality frame indices must be consecutive from 0; "
                f"expected {expected_frame}, got {frame_number}."
            )

        valid_text = str(row["valid"]).strip().lower()
        if valid_text not in {"true", "false", "1", "0"}:
            raise ValueError(
                f"Invalid 'valid' value for camera transition {frame_number}: "
                f"{row['valid']!r}"
            )

        valid = valid_text in {"true", "1"}
        method = str(row["method"]).strip().lower()
        if frame_number > 0 and (not valid or method == "failed"):
            failed_frames.append(frame_number)

    return failed_frames


def cumulative_transform(test_frame, transforms, failed_frames):
    crossed_failures = [
        frame_number
        for frame_number in failed_frames
        if REFERENCE_FRAME < frame_number <= test_frame
    ]
    if crossed_failures:
        frames = ", ".join(str(frame) for frame in crossed_failures)
        raise RuntimeError(
            f"Validation from frame {REFERENCE_FRAME} to {test_frame} "
            f"crosses failed camera transition(s) into frame(s): {frames}. "
            "No interpolation or alignment score was produced."
        )

    cumulative = np.eye(3, dtype=np.float64)
    for frame_number in range(REFERENCE_FRAME + 1, test_frame + 1):
        step = np.asarray(transforms[frame_number], dtype=np.float64)
        if not np.isfinite(step).all():
            raise RuntimeError(
                f"Camera transform for transition into frame {frame_number} "
                "contains nonfinite values."
            )
        determinant = float(np.linalg.det(step))
        if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
            raise RuntimeError(
                f"Camera transform for transition into frame {frame_number} "
                "is singular."
            )

        # Hs[f] maps frame f to frame f-1; right multiplication composes
        # the adjacent mappings into a frame-f-to-frame-0 transform.
        cumulative = cumulative @ step
        if not np.isfinite(cumulative).all():
            raise RuntimeError(
                f"Cumulative camera transform became nonfinite at frame "
                f"{frame_number}."
            )

    if not np.isfinite(cumulative).all():
        raise RuntimeError("Cumulative camera transform contains nonfinite values.")
    determinant = float(np.linalg.det(cumulative))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise RuntimeError("Cumulative camera transform is singular.")

    return cumulative


def read_frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read video frame {index}.")
    return frame


def project_landmarks(current_points, transform):
    homogeneous_points = np.column_stack(
        (current_points, np.ones(len(current_points), dtype=np.float64))
    )
    mapped_points = homogeneous_points @ transform.T
    denominators = mapped_points[:, 2]
    if (
        not np.isfinite(mapped_points).all()
        or np.any(np.abs(denominators) <= 1e-12)
    ):
        raise RuntimeError(
            "Transformation maps one or more selected landmarks to an "
            "invalid projective point; no valid alignment score was produced."
        )

    predicted = cv2.perspectiveTransform(
        current_points.reshape(-1, 1, 2).astype(np.float64),
        transform,
    ).reshape(-1, 2)
    if not np.isfinite(predicted).all():
        raise RuntimeError(
            "Transformation produced nonfinite predicted coordinates; "
            "no valid alignment score was produced."
        )
    return predicted


def select_landmarks(reference, current, test_frame):
    height, width = reference.shape[:2]
    scale = DISPLAY_WIDTH / width
    display_height = round(height * scale)
    reference_small = cv2.resize(
        reference, (DISPLAY_WIDTH, display_height)
    )
    current_small = cv2.resize(
        current, (DISPLAY_WIDTH, display_height)
    )
    reference_points = []
    current_points = []
    waiting_reference = True

    def draw_interface():
        left = reference_small.copy()
        right = current_small.copy()

        for index, (x, y) in enumerate(reference_points):
            point = (round(x * scale), round(y * scale))
            cv2.circle(left, point, 5, (0, 0, 255), -1)
            cv2.putText(
                left, str(index + 1), (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        for index, (x, y) in enumerate(current_points):
            point = (round(x * scale), round(y * scale))
            cv2.circle(right, point, 5, (0, 0, 255), -1)
            cv2.putText(
                right, str(index + 1), (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        combined = np.hstack([left, right])
        header = np.zeros(
            (HEADER_HEIGHT, combined.shape[1], 3), dtype=np.uint8
        )
        cv2.putText(
            header, f"REFERENCE FRAME {REFERENCE_FRAME}", (15, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            header, f"TEST FRAME {test_frame}",
            (DISPLAY_WIDTH + 15, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        landmark_id = (
            len(reference_points) + 1
            if waiting_reference
            else len(reference_points)
        )
        instruction = (
            f"Landmark {landmark_id}: click LEFT reference"
            if waiting_reference
            else f"Landmark {landmark_id}: click matching point on RIGHT"
        )
        cv2.putText(
            header, instruction, (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            header, "Choose stationary pitch-line intersections, not players "
            "or arbitrary points along lines.",
            (15, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 120), 1
        )
        cv2.putText(
            header, "Click LEFT then RIGHT | U: undo | R: reset | "
            "ENTER: finish | ESC: cancel",
            (15, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1
        )
        cv2.putText(
            header,
            f"Complete pairs: {len(current_points)} (minimum {MIN_LANDMARKS})",
            (15, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1
        )
        return np.vstack([header, combined])

    def mouse_callback(event, x, y, flags, userdata):
        nonlocal waiting_reference
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        image_y = y - HEADER_HEIGHT
        if (
            x < 0
            or x >= DISPLAY_WIDTH * 2
            or image_y < 0
            or image_y >= display_height
        ):
            return

        if waiting_reference:
            if x >= DISPLAY_WIDTH:
                print("Select the reference point in the LEFT image first.")
                return
            original_x = min(max(x / scale, 0.0), width - 1.0)
            original_y = min(max(image_y / scale, 0.0), height - 1.0)
            reference_points.append((original_x, original_y))
            waiting_reference = False
            print(f"Reference landmark {len(reference_points)} selected.")
        else:
            if x < DISPLAY_WIDTH:
                print("Select the matching point in the RIGHT image now.")
                return
            original_x = min(max((x - DISPLAY_WIDTH) / scale, 0.0), width - 1.0)
            original_y = min(max(image_y / scale, 0.0), height - 1.0)
            current_points.append((original_x, original_y))
            waiting_reference = True
            print(f"Matching landmark {len(current_points)} selected.")

    window = "Pitch Landmark Validation"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse_callback)
    print("\nPITCH LANDMARK VALIDATION")
    print("-" * 50)
    print("Click the same stationary pitch-line intersection in each frame.")
    print("Avoid players and arbitrary points along lines.")
    print("U = undo, R = reset, ENTER = finish, ESC = cancel.")

    try:
        while True:
            cv2.imshow(window, draw_interface())
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10):
                if not waiting_reference:
                    print("Finish the current landmark pair first.")
                elif len(current_points) < MIN_LANDMARKS:
                    print(f"At least {MIN_LANDMARKS} complete pairs are required.")
                else:
                    break
            elif key in (ord("u"), ord("U")):
                if not waiting_reference:
                    reference_points.pop()
                    waiting_reference = True
                elif current_points:
                    current_points.pop()
                    reference_points.pop()
                elif reference_points:
                    reference_points.pop()
                print("Undo completed.")
            elif key in (ord("r"), ord("R")):
                reference_points.clear()
                current_points.clear()
                waiting_reference = True
                print("All landmarks cleared.")
            elif key == 27:
                raise RuntimeError("Landmark selection cancelled.")
    finally:
        cv2.destroyAllWindows()

    return (
        np.asarray(reference_points, dtype=np.float64),
        np.asarray(current_points, dtype=np.float64),
    )


def append_summary(summary_path, record):
    """Append only identified runs; legacy CSVs lack reliable run metadata."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != SUMMARY_FIELDS:
                raise ValueError(
                    f"Existing summary has an unexpected schema: {summary_path}"
                )
            records = list(reader)
        write_header = False
    else:
        records = []
        write_header = True

    with summary_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    records.append(record)
    return records


def save_summary_chart(records, chart_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(
        records,
        key=lambda row: (int(row["test_frame"]), row["created_at"], row["run_id"]),
    )
    frames = [int(row["test_frame"]) for row in ordered]
    medians = [float(row["median_error_px"]) for row in ordered]
    maxima = [float(row["max_error_px"]) for row in ordered]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(frames, medians, marker="o", label="Median error")
    axis.plot(frames, maxima, marker="s", label="Maximum error")
    axis.set_title("Pitch landmark alignment error by test frame")
    axis.set_xlabel("Test frame number")
    axis.set_ylabel("Alignment error (pixels)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    frame_counts = {frame: frames.count(frame) for frame in set(frames)}
    duplicate_offsets = {}
    for row in ordered:
        frame = int(row["test_frame"])
        if frame_counts[frame] > 1:
            offset = duplicate_offsets.get(frame, 0)
            axis.annotate(
                row["run_id"][-6:],
                (frame, float(row["median_error_px"])),
                xytext=(4, 5 + offset * 9),
                textcoords="offset points",
                fontsize=7,
            )
            duplicate_offsets[frame] = offset + 1

    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def save_run(
    test_frame, reference, current, reference_points, current_points, predicted,
    errors
):
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    run_directory = OUTPUT / f"frame_{test_frame:04d}" / f"run_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)

    detailed_path = run_directory / "landmark_errors.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "reference_frame",
            "test_frame",
            "landmark_id",
            "reference_x",
            "reference_y",
            "current_x",
            "current_y",
            "predicted_reference_x",
            "predicted_reference_y",
            "error_px",
        ])
        for index, error in enumerate(errors):
            writer.writerow([
                REFERENCE_FRAME,
                test_frame,
                index + 1,
                *reference_points[index],
                *current_points[index],
                *predicted[index],
                float(error),
            ])

    selected_path = run_directory / "selected_points.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "reference_frame",
            "test_frame",
            "landmark_id",
            "reference_x",
            "reference_y",
            "current_x",
            "current_y",
        ])
        for index in range(len(reference_points)):
            writer.writerow([
                REFERENCE_FRAME,
                test_frame,
                index + 1,
                *reference_points[index],
                *current_points[index],
            ])

    visual = _draw_alignment_image(
        reference, test_frame, reference_points, predicted, errors
    )
    image_path = run_directory / "landmark_alignment.jpg"
    if not cv2.imwrite(str(image_path), visual):
        raise OSError(f"Could not save alignment image: {image_path}")

    selected_visual = _draw_selected_points_image(
        reference,
        current,
        test_frame,
        reference_points,
        current_points,
    )
    selected_image_path = run_directory / "landmark_selected_points.jpg"
    if not cv2.imwrite(str(selected_image_path), selected_visual):
        raise OSError(
            f"Could not save selected-points image: {selected_image_path}"
        )

    record = {
        "run_id": run_id,
        "created_at": created_at,
        "reference_frame": REFERENCE_FRAME,
        "test_frame": test_frame,
        "landmark_count": len(errors),
        "mean_error_px": f"{float(np.mean(errors)):.6f}",
        "median_error_px": f"{float(np.median(errors)):.6f}",
        "max_error_px": f"{float(np.max(errors)):.6f}",
        "p90_error_px": f"{float(np.percentile(errors, 90)):.6f}",
        "run_directory": str(run_directory),
    }
    summary_path = OUTPUT / "validation_summary.csv"
    records = append_summary(summary_path, record)
    save_summary_chart(records, OUTPUT / "validation_error_trends.png")
    return run_directory, record


def _put_labeled_text(image, text, origin, color, scale=0.55, thickness=2):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
    )


def _draw_alignment_image(
    reference, test_frame, reference_points, predicted, errors
):
    visual = reference.copy()
    height, width = visual.shape[:2]
    has_off_image_prediction = False

    for index, (actual, estimate, error) in enumerate(
        zip(reference_points, predicted, errors)
    ):
        actual_xy = tuple(np.rint(actual).astype(int))
        estimate_off_image = (
            estimate[0] < 0
            or estimate[0] >= width
            or estimate[1] < 0
            or estimate[1] >= height
        )
        has_off_image_prediction = (
            has_off_image_prediction or estimate_off_image
        )
        predicted_xy = (
            int(np.clip(np.rint(estimate[0]), 0, width - 1)),
            int(np.clip(np.rint(estimate[1]), 0, height - 1)),
        )
        cv2.line(visual, actual_xy, predicted_xy, (0, 255, 255), 2)
        cv2.circle(visual, actual_xy, 7, (0, 255, 0), -1)
        cv2.circle(visual, predicted_xy, 7, (0, 0, 255), -1)
        _put_labeled_text(
            visual,
            f"R{index + 1}",
            (actual_xy[0] + 10, actual_xy[1] - 10),
            (0, 255, 0),
        )
        _put_labeled_text(
            visual,
            f"P{index + 1}",
            (predicted_xy[0] + 10, predicted_xy[1] + 20),
            (0, 0, 255),
        )
        midpoint = (
            (actual_xy[0] + predicted_xy[0]) // 2,
            (actual_xy[1] + predicted_xy[1]) // 2,
        )
        _put_labeled_text(
            visual,
            f"{float(error):.1f} px",
            (midpoint[0] + 6, midpoint[1] - 6),
            (255, 255, 255),
            scale=0.48,
            thickness=1,
        )

    cv2.putText(
        visual,
        f"Frame {REFERENCE_FRAME} vs {test_frame}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    legend = np.zeros((74, width, 3), dtype=np.uint8)
    _put_labeled_text(
        legend, "R = reference", (20, 21),
        (0, 255, 0), scale=0.5, thickness=1
    )
    _put_labeled_text(
        legend, "P = transformed prediction", (20, 43),
        (0, 0, 255), scale=0.5, thickness=1
    )
    cv2.line(
        legend, (20, 62), (48, 62),
        (0, 255, 255), 2
    )
    _put_labeled_text(
        legend, "matching pair", (56, 67),
        (255, 255, 255), scale=0.5, thickness=1
    )
    if has_off_image_prediction:
        cv2.putText(
            visual,
            "Some predictions are off-image; markers are shown at the edge.",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    return np.vstack([visual, legend])


def _draw_selected_points_image(
    reference, current, test_frame, reference_points, current_points
):
    height, width = reference.shape[:2]
    if current.shape[:2] != (height, width):
        raise ValueError(
            "Reference and test frame dimensions do not match for visualization."
        )

    left = reference.copy()
    right = current.copy()
    for index, (reference_point, current_point) in enumerate(
        zip(reference_points, current_points), start=1
    ):
        reference_xy = tuple(np.rint(reference_point).astype(int))
        current_xy = tuple(np.rint(current_point).astype(int))
        cv2.circle(left, reference_xy, 7, (0, 255, 0), -1)
        cv2.circle(right, current_xy, 7, (0, 255, 0), -1)
        _put_labeled_text(
            left, f"R{index}", (reference_xy[0] + 10, reference_xy[1] - 10),
            (0, 255, 0)
        )
        _put_labeled_text(
            right, f"T{index}", (current_xy[0] + 10, current_xy[1] - 10),
            (0, 255, 0)
        )

    header = np.zeros((54, width * 2, 3), dtype=np.uint8)
    cv2.putText(
        header, f"REFERENCE FRAME {REFERENCE_FRAME}", (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    cv2.putText(
        header, f"TEST FRAME {test_frame}", (width + 12, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    cv2.putText(
        header, "R and T labels share the same landmark ID",
        (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1
    )
    return np.vstack([header, np.hstack([left, right])])


def main():
    parser, args = parse_args()
    if not VIDEO.is_file():
        parser.error(f"Video file not found: {VIDEO}")

    try:
        transforms = load_transforms(TRANSFORMS)
        capture = cv2.VideoCapture(str(VIDEO))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO}")

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError("Video reports no readable frames.")
            if args.test_frame < 1 or args.test_frame >= frame_count:
                raise ValueError(
                    f"Test frame {args.test_frame} is out of range; "
                    f"valid test frames are 1 through {frame_count - 1}."
                )
            failed_frames = load_transition_quality(
                QUALITY, frame_count, len(transforms)
            )
            transform = cumulative_transform(
                args.test_frame, transforms, failed_frames
            )
            reference = read_frame(capture, REFERENCE_FRAME)
            current = read_frame(capture, args.test_frame)
        finally:
            capture.release()

        if current.shape[:2] != reference.shape[:2]:
            raise RuntimeError("Reference and test frame dimensions do not match.")

        reference_points, current_points = select_landmarks(
            reference, current, args.test_frame
        )
        predicted = project_landmarks(current_points, transform)

        errors = np.linalg.norm(predicted - reference_points, axis=1)
        if not np.isfinite(errors).all():
            raise RuntimeError(
                "Alignment errors are nonfinite; no valid score was produced."
            )

        print("\nLANDMARK ALIGNMENT RESULTS")
        print("-" * 50)
        for index, error in enumerate(errors):
            print(f"Landmark {index + 1}: {error:.2f} px")
        print(f"\nMean error: {np.mean(errors):.2f} px")
        print(f"Median error: {np.median(errors):.2f} px")
        print(f"Maximum error: {np.max(errors):.2f} px")
        print(f"90th-percentile error: {np.percentile(errors, 90):.2f} px")

        run_directory, record = save_run(
            args.test_frame,
            reference,
            current,
            reference_points,
            current_points,
            predicted,
            errors,
        )
        print(f"\nRun ID: {record['run_id']}")
        print(f"Saved run: {run_directory}")
        print(f"Updated summary: {OUTPUT / 'validation_summary.csv'}")
        print(f"Updated chart: {OUTPUT / 'validation_error_trends.png'}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
