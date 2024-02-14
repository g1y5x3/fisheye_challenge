import io, os, cv2, wandb
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from utils import bounding_boxes 

WANDB = os.getenv("WANDB", False)

if __name__ == "__main__":
  if WANDB:
    run = wandb.init(project="yolov8-fisheye", name="yolov8x_predict_test_128")
    table = wandb.Table(columns=["ID", "Image"])
  
  data_dir = '../dataset/Fisheye8K_all_including_train/test/images/'
  label_dir = '../dataset/Fisheye8K_all_including_train/test/labels/'
  sources = [data_dir+img for img in os.listdir(data_dir)]
  print(f"Total data for inference {len(sources)}")
  
  model = YOLO('yolov8x.pt')
  class_coco = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'} 
  classid_fisheye = [5, 3, 2, 0, 7]
  
  #for i in range(len(sources)//128+1):
  for i in range(1):
    # starting and ending indices for each batch
    start = i*128
    end = (i+1)*128 if i<=20 else -1 
    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, conf=0.0, iou=0.7, stream=True)
    for result in results:
      img_id = result.path.rsplit('/',1)[-1]
      print(img_id)

      # TODO: do I need to save this or just use boxes to retrieve value for benchmarks calculation?
      result.save_txt("results/" + img_id.replace(".png", ".txt"))

      # Load the groundtruth for plotting comparison [x, y, width, height]
      boxes_gt = []
      with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
        boxes_gt = file.readlines()
      
      if WANDB:
        box_img = bounding_boxes(result.orig_img, result.boxes, boxes_gt, class_coco, classid_fisheye)
        table.add_data(img_id, box_img)

    # compute benchmarks against the groundtruth
    
  if WANDB:
    run.log({"Table" : table})
    run.finish()
