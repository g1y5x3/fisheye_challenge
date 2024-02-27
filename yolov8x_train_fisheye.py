"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import json, wandb, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from utils import get_image_id
def __init__(self, p=1.0):
  """Initialize the transform object for YOLO bbox formatted params."""
  self.p = p
  self.transform = None
  prefix = colorstr("albumentations: ")
  try:
    import albumentations as A

    # check_version(A.__version__, "1.0.3", hard=True)  # version requirement

    # Transforms
    T = [
      A.ToGray(p=0.01),
      A.CLAHE(p=0.01),
      A.RandomBrightnessContrast(p=0.0),
      A.RandomGamma(p=0.0),
      A.ImageCompression(quality_lower=75, p=0.0),
    ]
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
  except ImportError:  # package not installed, skip
    pass
  except Exception as e:
    LOGGER.info(f"{prefix}{e}")


# the default json was saved with "file_name" instead, saving as "image_id" makes it easier to 
# compute benchmarks # with cocoapi
def save_eval_json_with_id(validator):
  if not validator.training:
    pred_dir = "results/yolo_predictions.json"
    for pred in validator.jdict:
      pred["image_id"] = get_image_id(pred["image_id"])

    with open(pred_dir, "w") as f:
      LOGGER.info(f"Saving {pred_dir}...")
      json.dump(validator.jdict, f)

    artifact = wandb.Artifact(type="results", name=f"run_{wandb.run.id}_results")
    artifact.add_file(local_path=pred_dir)
    wandb.run.log_artifact(artifact)

    anno_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json"
    anno = COCO(anno_dir)
    pred = anno.loadRes(pred_dir)
    fisheye_eval = COCOeval(anno, pred, "bbox")
    print(fisheye_eval.params.areaRng)
    fisheye_eval.evaluate()
    fisheye_eval.accumulate()
    fisheye_eval.summarize()
      
    # log the mAP50-95 standard from the challenge
    wandb.run.log({"metrics/mAP50-95(maxDetx100)": fisheye_eval.stats[0]}, validator.args.epochs)

if __name__ == "__main__":

  Albumentations.__init__ = __init__

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1,  help="batch size")
  parser.add_argument('-epoch',   type=int, default=1,  help="number of epoch")
  parser.add_argument('-bs',      type=int, default=16, help="number of batches")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name',    type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(model="checkpoints/yolov8x.pt", data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, imgsz=1280,
                    project=args.project, name=args.name,
                    val=True, save_json=True,
                    exist_ok=True, # overwrite the existing dir
                    close_mosaic=0, # completely disable mosaic
                    degrees=0.1, translate=0.1, scale=0.0, shear=0.0, 
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=0.0, mixup=0.0)

  trainer = DetectionTrainer(overrides=train_args)
  trainer.add_callback("on_val_end", save_eval_json_with_id)
  trainer.train()

