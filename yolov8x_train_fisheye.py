import wandb
from ultralytics import YOLO

wandb.init(project="fisheye-challenge", name="yolov8x_train")

model = YOLO('yolov8x.pt') # model was pretrained on COCO dataset

results = model.train(data="fisheye.yaml", epochs=1, imgsz=640)

wandb.finish()
