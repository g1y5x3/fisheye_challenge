"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import torch, json, wandb, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.nn.tasks import DetectionModel
from utils import get_image_id

# libraries that are monkey patched
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from yolov8_monkey_patches import albumentation_init, load_model_custom, get_flops_pass, parse_dcn_model  

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

  # monkey patches
  Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="batch size")
  parser.add_argument('-frac', type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch', type=int, default=1, help="number of epoch")
  parser.add_argument('-bs', type=int, default=16, help="number of batches")
  parser.add_argument('-conf', type=float, default=0.001, help="confidence threshold")
  parser.add_argument('-iou', type=float, default=0.7, help="intersection of union")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name', type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(project=args.project, name=args.name, model=args.model, data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, fraction=args.frac, imgsz=1280,
                    exist_ok=True,
                    conf=args.conf, iou=args.iou,
                    optimizer="auto", seed=0,
                    box=7.5, cls=0.5, dfl=1.5,
                    lr0=0.01,
                    close_mosaic=0,
                    degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=0.0, mixup=0.0,
                    deterministic=True, verbose=True,
                    pretrained=True)

  trainer = DetectionTrainer(overrides=train_args)
  #trainer.add_callback("on_val_end", save_eval_json_with_id)
  trainer.train()

