import os, cv2, torch, json, wandb, argparse
import numpy as np
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO
from ultralytics.utils import ops
from utils import bounding_boxes_pred, get_image_id
from ultralytics.models.yolo.detect.train import DetectionTrainer
from yolov8_monkey_patches import load_model_custom, parse_dcn_model, get_flops_pass

if __name__ == "__main__":
  # monkey patches
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-conf', type=float, default=0.5, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.5,  help="number of workers")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="batch size")
  args = parser.parse_args()

  config = {"conf": args.conf,
            "iou" : args.iou}
  
  run = wandb.init(project="fisheye-challenge", name="yolov8x-dcn-1280-lr2e-5-mosaic_pred", config=config)

  art = run.use_artifact("g1y5x3/fisheye-challenge/run_dba333jb_model:best")
  art_dir = art.download()

  data_dir = "/workspace/FishEye8k/test_images/"
  files = [f for f in os.listdir(data_dir)]
  sources = [data_dir+img for img in files]
  print(f"Total data for inference {len(sources)}")

  model = YOLO(f"{art_dir}/best.pt") # model was trained on COCO dataset

  result_json = []
  result_dict = []
  for i in range(len(sources)//128+1):
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    results = model.predict(sources[start:end], imgsz=1280, 
                            conf=config["conf"], iou=config["iou"], 
                            stream=False, verbose=True)

    for result in results:
      image_name = result.path.rsplit('/',1)[-1]
      
      bboxes = ops.xyxy2ltwh(result.boxes.xyxy.cpu().numpy())
      conf = result.boxes.conf.cpu().numpy()
      cls = result.boxes.cls.cpu().numpy()
      for cat, score, box in zip(cls, conf, bboxes.tolist()):
        result_json.append(
          {
            "image_id": get_image_id(image_name),
            "category_id": int(cat),
            "bbox": [float(x) for x in box],
            "score": float(score)
          }
        )
        result_dict.append(
          {
            "image_id": image_name,
            "category_id": int(cat),
            "bbox": [float(x) for x in box],
            "score": float(score)
          }
        )

  with open("results/yolov8x_pred_id_fisheye.json", "w") as f:
    json.dump(result_json, f)

  with open("results/yolov8x_pred_name_fisheye.json", "w") as f:
    json.dump(result_dict, f)

  artifact = wandb.Artifact(type="results", name=f"run_{run.id}_results")
  artifact.add_file(local_path="results/yolov8x_pred_id_fisheye.json")
  run.log_artifact(artifact)

  table = wandb.Table(columns=["ID", "Image"])

  # Initialize the result calculation
  class_name = {0: 'bus', 1: 'bike', 2: 'car', 3: 'pedestrian', 4: 'truck'}

  pred_dir = "results/yolov8x_pred_name_fisheye.json"
  with open(pred_dir) as f: 
    preds = json.load(f)

  for img_name in files:
    print(img_name)
    
    bboxes = np.array([pred["bbox"] for pred in result_dict if pred["image_id"] == img_name])
    if len(bboxes) > 0:
      bboxes = ops.ltwh2xyxy(bboxes)
      confs = np.array([pred["score"] for pred in result_dict if pred["image_id"] == img_name])
      cls = np.array([pred["category_id"] for pred in result_dict if pred["image_id"] == img_name])
      pred_array = np.concatenate((bboxes, confs[:, np.newaxis], cls[:, np.newaxis]), axis=1)
    else:
      pred_array = np.array([])

    img = cv2.imread(data_dir + img_name)
    box_img = bounding_boxes_pred(img, pred_array.tolist(), class_name)
    table.add_data(img_name, box_img)

  run.log({"Table": table})
  run.finish()
  
