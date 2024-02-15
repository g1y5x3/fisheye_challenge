import wandb, argparse
from ultralytics import YOLO

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  args = parser.parse_args()

  arg = argparse
  wandb.init(project="fisheye-challenge", name="yolov8x_train")
  
  devices = args.devices
  model = YOLO('yolov8x.pt') # model was pretrained on COCO dataset
  
  # TODO: too much abstracted details
  results = model.train(data="fisheye.yaml", device=[i for i in range(devices)], epochs=1, batch=16, imgsz=640)
  
  wandb.finish()
