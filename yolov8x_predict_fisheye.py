# %%
import io, os, cv2, wandb
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

data_dir = '../dataset/Fisheye8K_all_including_train/test/images/'
sources = [data_dir+img for img in os.listdir(data_dir)[:5]]
print(f"Total data for inference {len(sources)}")

run = wandb.init(project="yolov8-fisheye", name="yolov8x_predict")

model = YOLO('yolov8x.pt')
results = model.predict(sources, save=False, imgsz=640, conf=0.5, stream=False)
names = model.names
print(f"class names {model.names}")

table = wandb.Table(columns=["ID", "Image"])
for result in results:
  img_id = result.path.rsplit('/',1)[-1]
  print(img_id)
  print(result.orig_img)

  box_img = wandb.Image(
    result.orig_img,
    boxes={
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
            "box_caption": names[int(box.cls.cpu().numpy()[0])],
          }
          for box in result.boxes
        ],
        "class_labels": names,
      }
    },
  )
  print(f"# of boxes {len(result.boxes)}")
  for box in result.boxes:
    print(box.xyxyn.cpu().numpy())
    print(box.xyxyn.cpu().numpy()[0])
    print(box.cls.cpu().numpy())
  table.add_data(img_id, box_img)
  
run.log({"Table" : table})
run.finish()
