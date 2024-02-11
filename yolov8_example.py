from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')

# Validate the model
metircs = model.val()
metircs.box.map