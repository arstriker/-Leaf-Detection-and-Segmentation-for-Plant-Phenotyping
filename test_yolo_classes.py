import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Assuming there is a test image we can use. I will just print the classes to check.
names = model.names
print({k: v for k, v in names.items() if 'plant' in v.lower() or 'leaf' in v.lower() or 'tree' in v.lower()})

