# %%
import io, os, cv2, wandb
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# %%
data_dir = '/home/iris/yg5d6/Workspace/ai_city/Fisheye8K_all_including_train/test/images/'
sources = [data_dir+img for img in os.listdir(data_dir)]
print(f"total data for inference {len(sources)}")

# %%
run = wandb.init(project="yolov8-fisheye", name="yolov8x_predict")
predictions_table = wandb.Table(columns=["filename", "prediction"])

model = YOLO('yolov8x.pt')
results_data = []
for source in sources:
  result = model.predict(source, save=False, imgsz=640, conf=0.5, stream=False)
  image = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.axis('off')
  plt.title(result[0].path.rsplit('/',1)[-1])
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  results_data.append([result[0].path.rsplit('/',1)[-1],
                       wandb.Image(Image.open(buf))])
results_table = wandb.Table(data=results_data, columns=["filename", "prediction"])
run.log({"prediction_table" : results_table})
run.finish()
