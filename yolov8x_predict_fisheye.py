import io, os, cv2, wandb
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

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
  class_coco = {0: 'person',
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck'} 
  class_fisheye = {0: 'bus',
                   1: 'bike',
                   2: 'car',
                   3: 'pedestrian',
                   4: 'truck'} 
  
  #for i in range(len(sources)//128+1):
  for i in range(1):
    start = i*128
    end = (i+1)*128 if i<=20 else -1 
    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, conf=0.5, stream=True)
    for result in results:
      img_id = result.path.rsplit('/',1)[-1]
      print(img_id)

      # TODO: do I need to save this or just use boxes to retrieve value for benchmarks calculation?
      result.save_txt("results/" + img_id.replace(".png", ".txt"))

      # Load the groundtruth for plotting comparison [x, y, width, height]
      boxes_gt = []
      with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
        boxes_gt = file.readlines()
      print(boxes_gt) 

      if WANDB:
        # TODO: unifying the box format to make it easier to compute the benchmarks
        box_img = wandb.Image(
          result.orig_img,
          boxes={
            "groundtruth": {
              "box_data": [
                {
                  "position": {
                      "middle": [float(box.split()[1]), float(box.split()[2])],
                      "width" : float(box.split()[3]),
                      "height": float(box.split()[4]),
                  },
                  "class_id": int(box.split()[0]),
                  "box_caption": class_fisheye[int(box.split()[0])],
                }
                for box in boxes_gt
              ],
              "class_labels": class_fisheye,
            },
            "prediction": {
              "box_data": [
                {
                  "position": {
                    "minX": float(box.xyxyn.cpu().numpy()[0][0]),
                    "minY": float(box.xyxyn.cpu().numpy()[0][1]),
                    "maxX": float(box.xyxyn.cpu().numpy()[0][2]),
                    "maxY": float(box.xyxyn.cpu().numpy()[0][3]),
                  },
                  "class_id": int(box.cls.cpu().numpy()[0]),
                  "box_caption": class_coco[int(box.cls.cpu().numpy()[0])],
                }
                for box in result.boxes
              ],
              "class_labels": class_coco,
            }
          },
        )
        table.add_data(img_id, box_img)
    
  if WANDB:
    run.log({"Table" : table})
    run.finish()
