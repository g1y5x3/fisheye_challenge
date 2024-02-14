import io, os, cv2, torch, wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, Metric
from utils import bounding_boxes 

WANDB = os.getenv("WANDB", False)

if __name__ == "__main__":
  if WANDB:
    run = wandb.init(project="fisheye-challenge", name="yolov8x_predict_test_128")
    table = wandb.Table(columns=["ID", "Image"])
  
  data_dir  = '../dataset/Fisheye8K_all_including_train/test/images/'
  label_dir = '../dataset/Fisheye8K_all_including_train/test/labels/'
  sources = [data_dir+img for img in os.listdir(data_dir)]
  print(f"Total data for inference {len(sources)}")

  # For the convenience of confusion matrix, all labels are convered to 0 ~ 4, probably can write it in a cleaner way
  class_name = {0: 'person', 1: 'car', 2: 'motorcycle', 3: 'bus', 4: 'truck'} 
  classid_coco    = {0:0, 2:1, 3:2, 5:3, 7:4}   # {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
  classid_fisheye = {0:3, 1:2, 2:1, 3:0, 4:4}   # {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'} 
 
  model = YOLO('yolov8x.pt')
  conf_mat = ConfusionMatrix(5, conf=0.25, iou_thres=0.45, task="detect")
 
  #for i in range(len(sources)//128+1):
  for i in range(1):
    # starting and ending indices for each batch
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, conf=0.0, iou=0.7, stream=True)
    for result in results:
      img_id = result.path.rsplit('/',1)[-1]
      print(img_id)

      # Load the groundtruth for corresponding image - [x, y, width, height]
      with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
        boxes_gt_string = file.readlines()

      # Calculate benchmarks
      # NOTE: might be easier to do if read from json instead
      gt_boxes = torch.empty((len(boxes_gt_string),4))
      gt_cls = torch.empty(len(boxes_gt_string))
      for i, box in enumerate(boxes_gt_string):
        gt_cls[i] = classid_fisheye[int(box.split()[0])]
        gt_boxes[i,:] = ops.xywh2xyxy(torch.tensor([float(box.split()[1]), float(box.split()[2]), float(box.split()[3]), float(box.split()[4])]))
      print("groundtruth", gt_boxes.shape)
      print("groundtruth", gt_cls.shape)

      # TODO: apparently when you set conf threhold to 0, the total amount of bounding boxes is capped at 300, 
      # most likely the top 300 ones but need to make sure that's the exact criteria
      cls = torch.tensor([classid_coco[i] for i in result.boxes.cls.cpu().numpy()])
      predict_boxes = torch.cat((result.boxes.xyxyn.cpu(), result.boxes.conf.cpu().unsqueeze(1), cls.unsqueeze(1)), dim=1)
      print("prediction", predict_boxes.shape)

      conf_mat.process_batch(predict_boxes, gt_boxes, gt_cls)
      
      if WANDB:
        box_img = bounding_boxes(result.orig_img, result.boxes, boxes_gt_string, class_name, classid_coco, classid_fisheye)
        table.add_data(img_id, box_img)

    # compute benchmarks against the groundtruth
    print(conf_mat.matrix)
    
  if WANDB:
    run.log({"Table" : table})
    run.finish()
