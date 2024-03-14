from ultralytics import YOLO

model = YOLO('yolov8x-p2.yaml')

parser = argparse.ArgumentParser(description="yolov8x")
parser.add_argument('-model', type=str, default="yolov8x-p2.yaml", help="model file name")
args = parser.parse_args()
 
results = model.train(data=args.model, epochs=250, imgsz=1024)
