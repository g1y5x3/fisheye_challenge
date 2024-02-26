"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import json, wandb, argparse
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect.train import DetectionTrainer
from utils import get_image_id

# the default json was saved with "file_name" instead, saving as "image_id" makes it easier to compute benchmarks
# with cocoapi
def save_eval_json_with_id(validator, benchmark=True):
  if not validator.training:
    pred_dir = "results/yolo_predictions.json"
    for pred in validator.jdict:
      pred["image_id"] = get_image_id(pred["image_id"])

    with open(pred_dir, "w") as f:
      LOGGER.info(f"Saving {pred_dir}...")
      json.dump(validator.jdict, f)

    art = wandb.Artifact(type="results", name=f"run_{wandb.run.id}_model")

    if benchmark:
      from pycocotools.coco import COCO
      from pycocotools.cocoeval import COCOeval

      anno_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/test.json"
      anno = COCO(anno_dir)
      pred = anno.loadRes(pred_dir)
      fisheye_eval = COCOeval(anno, pred, "bbox")
      print(fisheye_eval.params.areaRng)
      fisheye_eval.evaluate()
      fisheye_eval.accumulate()
      fisheye_eval.summarize()
        
      # log the mAP50-95 standard from the challenge
      wandb.run.log({"metrics/mAP50-95(maxDetx100)": fisheye_eval.stats[0]}, validator.epoch+1)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1,  help="batch size")
  parser.add_argument('-epoch',   type=int, default=1,  help="number of epoch")
  parser.add_argument('-bs',      type=int, default=16, help="number of batches")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name',    type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(model="checkpoints/yolov8x.pt", data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, imgsz=640,
                    project=args.project, name=args.name,
                    val=True, save_json=True)
  trainer = DetectionTrainer(overrides=train_args)
  trainer.add_callback("on_val_end", save_eval_json_with_id)
  trainer.train()

