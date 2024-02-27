from ultralytics import YOLO

# Load a YOLOv8n model
model = YOLO('yolov8x.pt')

# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(data="/workspace/fisheye_challenge/fisheye.yaml", use_ray=True, gpu_per_trial=1)
