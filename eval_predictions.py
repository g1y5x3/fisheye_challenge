import io, os, cv2, json, torch, wandb, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops
from utils import bounding_boxes, FisheyeDetectionValidator

WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME", "internimage_eval" )
      
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
  class_name = {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'}
  fisheye_eval = FisheyeDetectionValidator()
  fisheye_eval.init_metrics(class_name)

  # Load ground truth and predictions
  gt_dir   = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json"
  with open(gt_dir) as f: gts = json.load(f)
  gt_img, gt_ann = gts["images"], gts["annotations"]

  image_list = [img["file_name"] for img in gt_img]

  pred_dir = "results/internimage_eval_fisheye.json"
  with open(pred_dir) as f: preds = json.load(f)

  for img in image_list:
    print(img)
    # get ground truth
    img_id = [gt["id"] for gt in gt_img if gt["file_name"] == img]
    bboxes = np.array([gt["bbox"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    bboxes = ops.ltwh2xyxy(bboxes)
    cls = np.array([gt["category_id"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    gt_array = torch.tensor(np.concatenate((bboxes, cls[:, np.newaxis]), axis=1))
    
    # get predictions
    bboxes = np.array([pred["bbox"] for pred in preds if pred["image_id"] == img])
    bboxes = ops.ltwh2xyxy(bboxes)
    confs = np.array([pred["score"] for pred in preds if pred["image_id"] == img])
    cls_convert = {0:3, 2:2, 3:1, 5:0, 7:4}  # only need this with raw model trained with coco
    cls = np.array([cls_convert[pred["category_id"]] for pred in preds if pred["image_id"] == img])
    pred_array = torch.tensor(np.concatenate((bboxes, confs[:, np.newaxis], cls[:, np.newaxis]), axis=1))

    fisheye_eval.update_metrics([pred_array], [gt_array])

  print(fisheye_eval.confusion_matrix.matrix)
  fisheye_eval.confusion_matrix.plot(save_dir="results", names=tuple(class_name.values()))
  stat = fisheye_eval.get_stats()
  print(fisheye_eval.get_desc())
  fisheye_eval.print_results()
    
  if WANDB:
    run.log(stat)
    run.log({"metrics/conf_mat(B)": wandb.Image("results/confusion_matrix_normalized.png")})
    #run.log({"Table": table})
    run.finish()

  #for i in range(len(sources)//128+1):
  #for i in range(1):
  #  start = i*128
  #  end = (i+1)*128 if i <= 20 else -1 

  #  results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, 
  #                          conf=config["model/conf"], iou=config["model/iou"], 
  #                          stream=False, verbose=True)
  #  print(results)

  #  preds, gts = [], []
  #  for result in results:
  #    print(result.tojson())
  #    img_id = result.path.rsplit('/',1)[-1]

  #    # Load the groundtruth for corresponding image - [x, y, width, height]
  #    with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
  #      boxes_gt_string = file.readlines()

  #    # convert both predictions and ground truths into the format to calculate benchmarks
  #    gt = torch.empty((len(boxes_gt_string), 5))
  #    for i_box, box in enumerate(boxes_gt_string):
  #      gt[i_box, :4] = ops.xywh2xyxy(torch.tensor([float(box.split()[1]),
  #                                                  float(box.split()[2]),
  #                                                  float(box.split()[3]),
  #                                                  float(box.split()[4])]))
  #      gt[i_box,  4] = classid_fisheye[int(box.split()[0])]
  #    gts.append(gt)

  #    # NOTE: apparently when you set conf threhold to 0, the total amount of bounding boxes is capped at 300, 
  #    # most likely the top 300 ones but need to make sure that's the exact criteria
  #    cls = torch.tensor([classid_coco[i] for i in result.boxes.cls.cpu().numpy()])
  #    pred = torch.cat((result.boxes.xyxyn.cpu(), result.boxes.conf.cpu().unsqueeze(1), cls.unsqueeze(1)), dim=1)
  #    preds.append(pred)

  #    if WANDB:
  #      box_img = bounding_boxes(result.orig_img, result.boxes, 
  #                               boxes_gt_string, class_name, classid_coco, classid_fisheye)
  #      table.add_data(img_id, box_img)

  #  fisheye_eval.update_metrics(preds, gts)

  #print(fisheye_eval.confusion_matrix.matrix)
  #fisheye_eval.confusion_matrix.plot(save_dir="results", names=tuple(class_name.values()))
  #stat = fisheye_eval.get_stats()
  #print(fisheye_eval.get_desc())
  #fisheye_eval.print_results()
  #  
  #if WANDB:
  #  run.log(stat)
  #  run.log({"metrics/conf_mat(B)": wandb.Image("results/confusion_matrix_normalized.png")})
  #  run.log({"Table": table})
  #  run.finish()
