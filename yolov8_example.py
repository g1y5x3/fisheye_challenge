# the usege of this example script was to invoke all possible details of the library
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')

# Train a model
results = model.train(data='coco128.yaml', epochs=1, imgsz=640)

# Validate the model
metircs = model.val()
