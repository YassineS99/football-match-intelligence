import cv2
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = "data/input/sample.mp4"
OUTPUT_PATH = "data/output/tracked_full.mp4"
MAX_FRAMES = None  # set to None later to process the whole clip
PERSON_CLASS_ID = 0

model = YOLO("yolov8x.pt")

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
trace_annotator = sv.TraceAnnotator(trace_length=40, thickness=2)

frames = sv.get_video_frames_generator(VIDEO_PATH, end=MAX_FRAMES)
total = MAX_FRAMES if MAX_FRAMES else video_info.total_frames

with sv.VideoSink(OUTPUT_PATH, video_info) as sink:
    for frame in tqdm(frames, total=total, desc="Tracking"):
        # persist=True keeps track IDs between frames
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS_ID],
            conf=0.25,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated = frame.copy()
        if detections.tracker_id is not None and len(detections) > 0:
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

        sink.write_frame(annotated)

print("Saved:", OUTPUT_PATH)