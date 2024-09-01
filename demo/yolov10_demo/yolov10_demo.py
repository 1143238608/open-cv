import cv2
import supervision as sv
from ultralytics import YOLOv10
from PIL import Image

# Load a pretrained YOLOv10n model
# D:\code\OpencvPeoject\yolov10\runs\detect\train4\weights\best.pt
model = YOLOv10(r"D:\code\OpencvPeoject\yolov10\runs\detect\train4\weights\best.pt")

cap = cv2.VideoCapture(r'D:\code\OpencvPeoject\demo\yolov10_demo\test_1.mp4')

while True:
    ret, frame = cap.read()
    # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    corner_annotator = sv.BoxCornerAnnotator(corner_length=10, thickness=1, color=sv.Color(r=0, g=255, b=255))
    corner_image = corner_annotator.annotate(scene=frame, detections=detections)
    cv2.imshow('ssss', corner_image)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
