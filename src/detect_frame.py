import cv2
import supervision as sv
from ultralytics import YOLO

VIDEO_PATH = "data/input/sample.mp4"
OUTPUT_PATH = "data/output/frame_detections.jpg"
FRAME_INDEX = 100

# Downloads the weights automatically on first run
model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ok, frame = cap.read()
cap.release()

if not ok:
    raise SystemExit("Could not read the frame")

result = model(frame, conf=0.25, verbose=False)[0]
detections = sv.Detections.from_ultralytics(result)

print("Class names the model knows (first 10):", list(result.names.values())[:10])
print("Total detections:", len(detections))
for class_id in sorted(set(detections.class_id)):
    name = result.names[class_id]
    count = int((detections.class_id == class_id).sum())
    print(f"  {name}: {count}")

box_annotator = sv.BoxAnnotator()
annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
cv2.imwrite(OUTPUT_PATH, annotated)
print("Saved:", OUTPUT_PATH)