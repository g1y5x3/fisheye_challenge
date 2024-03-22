"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import argparse
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK

def load_model(self, cfg=None, weights=None, verbose=True):
  model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  weights, _ = attempt_load_one_weight("checkpoints/rtdetr-x.pt")
  model.load(weights)
  return model

if __name__ == "__main__":
  RTDETRTrainer.get_model = load_model

  parser = argparse.ArgumentParser(description="fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  parser.add_argument('-model', type=str, default="rtdetr-x.yaml", help="model file name")
  parser.add_argument('-imgsz', type=int, default=1280, help="image size used for training")
  parser.add_argument('-frac', type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch', type=int, default=1, help="number of epoch")
  parser.add_argument('-bs', type=int, default=16, help="number of batches")
  parser.add_argument('-wd', type=float, default=0.0005, help="weight decay")
  parser.add_argument('-conf', type=float, default=0.001, help="confidence threshold")
  parser.add_argument('-iou', type=float, default=0.5, help="intersection of union")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name', type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  #https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt
  train_args = dict(project=args.project, name=args.name, model=args.model, data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, imgsz=args.imgsz, fraction=args.frac,
                    exist_ok=True,
                    conf=args.conf, iou=args.iou,
                    lr0=2e-4, warmup_bias_lr=2e-4/3, weight_decay=args.wd,
                    optimizer="AdamW", seed=0,
                    box=7.5, cls=0.5, dfl=1.5,
                    close_mosaic=10,
                    degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=1.0, mixup=0.0,
                    deterministic=True, verbose=True,
                    pretrained=True)

  trainer = RTDETRTrainer(overrides=train_args)
  trainer.train()

