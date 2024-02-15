import io, os, cv2, torch, wandb, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops
from utils import bounding_boxes, FisheyeDetectionValidator

WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME", "yolov8x_eval" )
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-conf', type=float, default=0.25, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.7,  help="number of workers")
  args = parser.parse_args()

  config = {"model/conf": args.conf,
            "model/iou" : args.iou}

  if WANDB:
    run = wandb.init(project="fisheye-challenge", name=NAME, config=config)
    table = wandb.Table(columns=["ID", "Image"])
  
  data_dir  = '../dataset/Fisheye8K_all_including_train/test/images/'
  label_dir = '../dataset/Fisheye8K_all_including_train/test/labels/'
  sources = [data_dir+img for img in os.listdir(data_dir)]
  print(f"Total data for inference {len(sources)}")

  # For the convenience of confusion matrix, all labels are convered to 0 ~ 4, probably can write it in an easier way
  class_name = {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'} 
  classid_fisheye = {0:0, 1:1, 2:2, 3:3, 4:4}   # {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'} 
  classid_coco    = {0:3, 2:2, 3:1, 5:0, 7:4}   # {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
 
  model = YOLO('yolov8x.pt') # model was trained on COCO dataset
  fisheye_eval = FisheyeDetectionValidator()
  fisheye_eval.init_metrics(class_name) 
 
  for i in range(len(sources)//128+1):
    # starting and ending indices for each batch
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    # NOTE: the 'iou' here is used for NMS
    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, conf=config["model/conf"], iou=config["model/iou"], stream=True, verbose=True)

    preds, gts = [], []
    for result in results:
      img_id = result.path.rsplit('/',1)[-1]

      # Load the groundtruth for corresponding image - [x, y, width, height]
      with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
        boxes_gt_string = file.readlines()

      # convert both predictions and ground truths into the format to calculate benchmarks
      gt = torch.empty((len(boxes_gt_string), 5))
      for i_box, box in enumerate(boxes_gt_string):
        gt[i_box, :4] = ops.xywh2xyxy(torch.tensor([float(box.split()[1]), float(box.split()[2]), float(box.split()[3]), float(box.split()[4])]))
        gt[i_box,  4] = classid_fisheye[int(box.split()[0])]
      gts.append(gt)

      # NOTE: apparently when you set conf threhold to 0, the total amount of bounding boxes is capped at 300, 
      # most likely the top 300 ones but need to make sure that's the exact criteria
      cls = torch.tensor([classid_coco[i] for i in result.boxes.cls.cpu().numpy()])
      pred = torch.cat((result.boxes.xyxyn.cpu(), result.boxes.conf.cpu().unsqueeze(1), cls.unsqueeze(1)), dim=1)
      preds.append(pred)

      if WANDB:
        box_img = bounding_boxes(result.orig_img, result.boxes, boxes_gt_string, class_name, classid_coco, classid_fisheye)
        table.add_data(img_id, box_img)

    fisheye_eval.update_metrics(preds, gts)

  print(fisheye_eval.confusion_matrix.matrix)
  fisheye_eval.confusion_matrix.plot(save_dir="results", names=tuple(class_name.values()))
  stat = fisheye_eval.get_stats()
  print(fisheye_eval.get_desc())
  fisheye_eval.print_results()
    
  if WANDB:
    run.log(stat)
    run.log({"metrics/conf_mat(B)": wandb.Image("results/confusion_matrix_normalized.png")})
    run.log({"Table": table})
    run.finish()
