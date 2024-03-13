import os, cv2, torch, json, wandb, argparse
import numpy as np
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO
from ultralytics.utils import ops
#from ultralytics.data.augment import Albumentations
from utils import bounding_boxes, FisheyeDetectionValidator
from ultralytics.models.yolo.detect.train import DetectionTrainer
from yolov8_monkey_patches import albumentation_init, load_model_custom, parse_dcn_model, get_flops_pass

if __name__ == "__main__":
  # monkey patches
  #Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-conf', type=float, default=0.001, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.5,  help="number of workers")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="batch size")
  args = parser.parse_args()

  config = {"conf": args.conf,
            "iou" : args.iou}
  
  run = wandb.init(project="fisheye-challenge", name="yolov8x_eval", config=config)

  art = run.use_artifact("g1y5x3/fisheye-challenge/run_zypfjyhk_model:v0")
  art_dir = art.download()

  data_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images/"

  with open("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json") as f:
    images = json.load(f)
  files = images["images"]
  sources = [data_dir+img["file_name"] for img in files]
  print(f"Total data for inference {len(sources)}")

  model = YOLO(f"{art_dir}/best.pt") # model was trained on COCO dataset

  result_json = []
  for i in range(len(sources)//128+1):
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    results = model.predict(sources[start:end], imgsz=1280, 
                            conf=config["conf"], iou=config["iou"], 
                            stream=False, verbose=True)

    for result in results:
      image_id = result.path.rsplit('/',1)[-1]
      bboxes = ops.xyxy2ltwh(result.boxes.xyxy.cpu().numpy())
      conf = result.boxes.conf.cpu().numpy()
      cls = result.boxes.cls.cpu().numpy()
      for cat, score, box in zip(cls, conf, bboxes.tolist()):
        result_json.append(
          {
            "image_id": image_id,
            "category_id": int(cat),
            "bbox": [float(x) for x in box],
            "score": float(score)
          }
        )

  with open("results/yolov8x_eval_fisheye.json", "w") as f:
    json.dump(result_json, f)

  table = wandb.Table(columns=["ID", "Image"])

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

  for img_name in image_list:
    print(img_name)
    # get ground truth
    img_id = [gt["id"] for gt in gt_img if gt["file_name"] == img_name]
    bboxes = np.array([gt["bbox"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    bboxes = ops.ltwh2xyxy(bboxes)
    cls = np.array([gt["category_id"] for gt in gt_ann if gt["image_id"] == img_id[0]])
    gt_array = np.concatenate((bboxes, cls[:, np.newaxis]), axis=1)
    
    # get predictions
    bboxes = np.array([pred["bbox"] for pred in preds if pred["image_id"] == img_name])
    if len(bboxes) > 0:
      bboxes = ops.ltwh2xyxy(bboxes)
      confs = np.array([pred["score"] for pred in preds if pred["image_id"] == img_name])
      cls = np.array([pred["category_id"] for pred in preds if pred["image_id"] == img_name])
      pred_array = np.concatenate((bboxes, confs[:, np.newaxis], cls[:, np.newaxis]), axis=1)
    else:
      pred_array = np.array([])

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
    
  run.log(stat)
  run.log({"metrics/conf_mat(B)": wandb.Image("results/confusion_matrix_normalized.png")})
  run.log({"Table": table})
  run.finish()
  
