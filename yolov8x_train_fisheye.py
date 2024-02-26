"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import json, wandb, argparse
from utils import get_image_id
from ultralytics.models.yolo.detect.train import DetectionTrainer

# the default json was saved with "file_name" instead, saving as "image_id" makes it easier to compute benchmarks
# with cocoapi
def save_eval_json_with_id(validator):
  if not validator.training:
    pred_dir = "results/yolo_predictions.json"
    print("THIS IS VALIDATOR CALLBACK")
    print(validator.jdict[:10])
    for pred in validator.jdict:
      pred["image_id"] = get_image_id(pred["image_id"])

    print("AFTER CONVERSION")
    print(validator.jdict[:10])
    with open(pred_dir, "w") as f:
      json.dump(validator.jdict, f)

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

  ## convert the "image_id" from name to id for benchmarks
  #pred_dir = args.project + "/" + args.name + "/" +  "predictions.json"
  #print(pred_dir)

  #with open(pred_dir) as f:
  #  predictions = json.load(f)

  #for pred in predictions:
  #  pred["image_id"] = get_image_id(pred["image_id"])

  #with open(pred_dir, "w") as f:
  #  json.dump(predictions, f)
