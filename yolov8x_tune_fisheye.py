import torch, json, wandb, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.nn.tasks import DetectionModel
from utils import get_image_id

# libraries that are monkey patched
from ultralytics.utils import LOGGER, RANK, colorstr
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from yolov8_monkey_patches import get_flops_pass, parse_dcn_model  

def albumentation_init(self, p=1.0):
  """Initialize the transform object for YOLO bbox formatted params."""
  self.p = p
  self.transform = None
  prefix = colorstr("albumentations: ")
  try:
    import albumentations as A

    # check_version(A.__version__, "1.0.3", hard=True)  # version requirement

    # Transforms
    T = [
      A.RandomBrightnessContrast(p=0.01),
    ]
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
  except ImportError:  # package not installed, skip
    pass
  except Exception as e:
    LOGGER.info(f"{prefix}{e}")


def load_model_custom(self, cfg=None, weights=None, verbose=True):
  """Return a YOLO detection model."""
  name = cfg.split("_")[0]
  print(name)

  art = wandb.use_artifact(f"g1y5x3/fisheye-challenge/run_zh55zy10_model:best")
  art_dir = art.download()
  weights, _ = attempt_load_one_weight(f"{art_dir}/best.pt")

  #weights, _ = attempt_load_one_weight(f"checkpoints/{name}.pt")

  model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  print(f"config pretrained: {self.args.pretrained}")
  if weights and self.args.pretrained:
    model.load(weights)
  return model

if __name__ == "__main__":
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
  parser.add_argument('-epoch', type=int, default=10, help="number of epoch")
  parser.add_argument('-bs', type=int, default=16, help="number of batches")
  parser.add_argument('-wd', type=float, default=0.0005, help="weight decay")
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
                    degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=1.0, mixup=0.0,
                    deterministic=True, verbose=True,
                    pretrained=True)

  trainer = DetectionTrainer(overrides=train_args)
  trainer.train()
