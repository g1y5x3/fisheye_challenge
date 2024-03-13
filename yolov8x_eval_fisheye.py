import os, json, wandb, argparse
from ultralytics import YOLO
from ultralytics.utils import ops
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from yolov8_monkey_patches import albumentation_init, load_model_custom, parse_dcn_model, get_flops_pass

if __name__ == "__main__":
  # monkey patches
  Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-conf', type=float, default=0.5, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.5,  help="number of workers")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="batch size")
  args = parser.parse_args()

  config = {"model/conf": args.conf,
            "model/iou" : args.iou}
  
  run = wandb.init(project="artifacts-example")
  art = run.use_artifact("g1y5x3/fisheye-challenge/run_zypfjyhk_model:v0")
  art_dir = art.download()

  data_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images/"

  with open("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json") as f:
    images = json.load(f)
  files = images["images"]
  sources = [data_dir+img["file_name"] for img in files]
  print(f"Total data for inference {len(sources)}")

  model = YOLO(f"{art_dir}/best.pt") # model was trained on COCO dataset
  print(model) 

  #result_json = []
  ##for i in range(len(sources)//128+1):
  #for i in range(1):
  #  start = i*128
  #  end = (i+1)*128 if i <= 20 else -1 

  #  results = model.predict(sources[start:end], imgsz=1280, 
  #                          conf=config["model/conf"], iou=config["model/iou"], 
  #                          stream=False, verbose=True)

  #  for result in results:
  #    image_id = result.path.rsplit('/',1)[-1]
  #    bboxes = ops.xyxy2ltwh(result.boxes.xyxy.cpu().numpy())
  #    conf = result.boxes.conf.cpu().numpy()
  #    cls = result.boxes.cls.cpu().numpy()
  #    for cat, score, box in zip(cls, conf, bboxes.tolist()):
  #      result_json.append(
  #        {
  #          "image_id": image_id,
  #          "category_id": int(cat),
  #          "bbox": [float(x) for x in box],
  #          "score": float(score)
  #        }
  #      )

  #with open("results/yolov8x_eval_fisheye_128.json", "w") as f:
  #  json.dump(result_json, f)
