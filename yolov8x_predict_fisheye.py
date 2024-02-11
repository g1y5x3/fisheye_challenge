# %%
import io, os, cv2, wandb
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# %%
data_dir = 'dataset/Fisheye8K_all_including_train/test/images/'
sources = [data_dir+img for img in os.listdir(data_dir)[:100]]
print(f"total data for inference {len(sources)}")

# %%
run = wandb.init(project="yolov8-fisheye", name="yolov8x_predict")

predictions_table = wandb.Table(columns=["filename", "prediction"])

model = YOLO('yolov8x.pt')
results = model.predict(sources, save=False, imgsz=640, conf=0.5, stream=False)
results_data = []
for result in results:
  image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.axis('off')
  plt.title(result.path.rsplit('/',1)[-1])
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  results_data.append([result.path.rsplit('/',1)[-1],
                       wandb.Image(Image.open(buf))])
results_table = wandb.Table(data=results_data, columns=["filename", "prediction"])
print(results_table)

run.log({"prediction_table" : results_table})
run.finish()
