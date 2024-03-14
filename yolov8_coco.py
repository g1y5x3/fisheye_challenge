from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser(description="yolov8x")
parser.add_argument('-model', type=str, default="yolov8x-p2.yaml", help="model file name")
args = parser.parse_args()
 
model = YOLO(args.model)
results = model.train(data='coco.yaml', epochs=250, imgsz=1024)
