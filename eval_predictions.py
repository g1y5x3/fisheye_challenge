import os, cv2, json, torch, wandb, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics.utils import ops
from utils import bounding_boxes, FisheyeDetectionValidator

WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME", "yolov8x_eval" )
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="fisheye experiment evaluation")
  parser.add_argument('-conf', type=float, default=0.001, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.7,  help="number of workers")
  args = parser.parse_args()

  config = {"epochs": 0,
            "model/conf": args.conf,
            "model/iou" : args.iou}

  if WANDB:
    run = wandb.init(project="fisheye-challenge", name=NAME, config=config)
    table = wandb.Table(columns=["ID", "Image"])
  
  data_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images/"

  # Initialize the result calculation
  class_name = {0: 'bus', 1: 'bike', 2: 'car', 3: 'pedestrian', 4: 'truck'}
  fisheye_eval = FisheyeDetectionValidator()
  fisheye_eval.init_metrics(class_name)

  # Load ground truth and predictions
  gt_dir   = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json"
  with open(gt_dir) as f: gts = json.load(f)
  gt_img, gt_ann = gts["images"], gts["annotations"]

  image_list = [img["file_name"] for img in gt_img]

  pred_dir = "results/yolov8x_eval_fisheye.json"
  with open(pred_dir) as f: preds = json.load(f)

  for img_name in image_list[:128]:
    print(img_name)
    # get ground truth
    img_id = [gt["id"] for gt in gt_img if gt["file_name"] == img_name]
    bboxes = np.array([gt["bbox"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    bboxes = ops.ltwh2xyxy(bboxes)
    cls = np.array([gt["category_id"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    gt_array = np.concatenate((bboxes, cls[:, np.newaxis]), axis=1)
    #gt_array = torch.tensor(np.concatenate((bboxes, cls[:, np.newaxis]), axis=1))
    
    # get predictions
    bboxes = np.array([pred["bbox"] for pred in preds if pred["image_id"] == img_name])
    if len(bboxes) > 0:
      bboxes = ops.ltwh2xyxy(bboxes)
      confs = np.array([pred["score"] for pred in preds if pred["image_id"] == img_name])
      cls = np.array([pred["category_id"] for pred in preds if pred["image_id"] == img_name])
      pred_array = np.concatenate((bboxes, confs[:, np.newaxis], cls[:, np.newaxis]), axis=1)
      #pred_array = torch.tensor(np.concatenate((bboxes, confs[:, np.newaxis], cls[:, np.newaxis]), axis=1))
    else:
      pred_array = np.array([])

    if WANDB:
      img = cv2.imread(data_dir + img_name)
      box_img = bounding_boxes(img, pred_array.tolist(), gt_array.tolist(), class_name)
      table.add_data(img_name, box_img)

    fisheye_eval.update_metrics([torch.tensor(pred_array)], [torch.tensor(gt_array)])

  print("Confusion Matrix:")
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

