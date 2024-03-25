"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import os, torch, json, wandb, argparse, shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from utils import get_image_id

# libraries that are monkey patched
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import RANK
from yolov8_monkey_patches import albumentation_init, get_flops_pass, parse_dcn_model  

def load_model_custom(self, cfg=None, weights=None, verbose=True):
  # load the weights from the current best model
  name = cfg.split("_")[0]
  weights, _ = attempt_load_one_weight(f"checkpoints/best.pt")
  model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  print(f"config pretrained: {self.args.pretrained}")
  if weights and self.args.pretrained:
    model.load(weights)
  return model

if __name__ == "__main__":
  model = YOLO(f"checkpoints/best.pt")

  data_dir = "/workspace/FishEye8k/test_images/"
  files = [f for f in os.listdir(data_dir)]
  sources = [data_dir+img for img in files]
  print(f"Total data for inference {len(sources)}")

  for i in range(len(sources)//128+1):
    start = i*128
    end = (i+1)*128 if i <= 20 else -1

    results = model.predict(sources[start:end], imgsz=1280,
                            conf=0.5, iou=0.5,
                            stream=False, verbose=True, save_txt=True)

  # copy both images and labels to the training directory
  for file in os.listdir("/workspace/FishEye8k/test_images"):
    print(file)
    shutil.copy("/workspace/FishEye8k/test_images"+"/"+file,
                "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/images"+"/"+file)

  for file in os.listdir("/usr/src/ultralytics/runs/detect/predict/labels"):
    print(file)
    shutil.copy("/usr/src/ultralytics/runs/detect/predict/labels"+"/"+file,
                "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/labels"+"/"+file)

  # monkey patches
  Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="model file name")
  parser.add_argument('-imgsz', type=int, default=1280, help="image size used for training")
  parser.add_argument('-frac', type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch', type=int, default=1, help="number of epoch")
  parser.add_argument('-bs', type=int, default=16, help="number of batches")
  parser.add_argument('-wd', type=float, default=0.0025, help="weight decay")
  parser.add_argument('-conf', type=float, default=0.001, help="confidence threshold")
  parser.add_argument('-iou', type=float, default=0.5, help="intersection of union")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name', type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(project=args.project, name=args.name, model=args.model, data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, imgsz=args.imgsz, fraction=args.frac,
                    exist_ok=True,
                    conf=args.conf, iou=args.iou,
                    lr0=2e-5, warmup_bias_lr=2e-5/3, weight_decay=args.wd,
                    optimizer="AdamW", seed=0,
                    box=7.5, cls=0.5, dfl=1.5,
                    close_mosaic=10,
                    degrees=0, translate=0.1, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=1.0, mixup=0.0,
                    deterministic=True, verbose=True,
                    pretrained=True)

  trainer = DetectionTrainer(overrides=train_args)
  trainer.train()
