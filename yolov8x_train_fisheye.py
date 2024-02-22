"""
python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32
"""
import wandb, argparse
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1,  help="batch size")
  parser.add_argument('-epoch',   type=int, default=1,  help="number of epoch")
  parser.add_argument('-bs',      type=int, default=16, help="number of batches")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]
  train_args = dict(model="checkpoints/yolov8x.pt", data="fisheye.yaml", device=device, epochs=args.epoch, batch=args.bs, imgsz=640)
  trainer = DetectionTrainer(overrides=train_args)
  trainer.train()
