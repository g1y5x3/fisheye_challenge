import os, json, argparse
from ultralytics import YOLO
from ultralytics.utils import ops

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-conf', type=float, default=0.001, help="batch size")
  parser.add_argument('-iou',  type=float, default=0.7,  help="number of workers")
  args = parser.parse_args()

  config = {"model/conf": args.conf,
            "model/iou" : args.iou}
  
  data_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images/"
  with open("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json") as f:
    images = json.load(f)
  files = images["images"]
  sources = [data_dir+img["file_name"] for img in files]
  print(f"Total data for inference {len(sources)}")

  model = YOLO('checkpoints/yolov8x.pt') # model was trained on COCO dataset
 
  result_json = []
  #for i in range(len(sources)//128+1):
  for i in range(1):
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, 
                            conf=config["model/conf"], iou=config["model/iou"], 
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

  with open("results/yolov8x_eval_fisheye_128.json", "w") as f:
    json.dump(result_json, f)
