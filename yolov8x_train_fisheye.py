"""
if __name__ == "__main__":
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    print(cfg)
    print(overrides)
    trainer = DetectionTrainer(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
import wandb, argparse
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  args = parser.parse_args()
  
  devices = args.devices
  
  # TODO: too much abstracted details
  args = dict(model="yolov8x.pt", data="fisheye.yaml", device=[i for i in range(devices)], epochs=1, batch=32, imgsz=640)
  trainer = DetectionTrainer(overrides=args)
  trainer.train()
  
