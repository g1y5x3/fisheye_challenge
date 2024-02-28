"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import torch, json, wandb, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.nn.modules import Detect, Segment, Pose, OBB
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer

from utils import get_image_id

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

class DetectionModel(BaseModel):
    """YOLOv8 detection model."""
	# model, input channels, number of classes
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)

# monkey patching to bypass the unwanted albumentation
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
      A.RandomBrightnessContrast(p=0.01),
      A.RandomGamma(p=0.01),
    ]
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
  except ImportError:  # package not installed, skip
    pass
  except Exception as e:
    LOGGER.info(f"{prefix}{e}")

if __name__ == "__main__":

  Albumentations.__init__ = __init__

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int,   default=1,   help="batch size")
  parser.add_argument('-frac',    type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch',   type=int,   default=1,   help="number of epoch")
  parser.add_argument('-bs',      type=int,   default=16,  help="number of batches")
  parser.add_argument('-project', type=str,   default="fisheye-challenge", help="project name")
  parser.add_argument('-name',    type=str,   default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(model="checkpoints/yolov8x.pt", data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, fraction=args.frac, imgsz=1280,
                    project=args.project, name=args.name,
                    val=True, save_json=True, exist_ok=True,
                    close_mosaic=0, # completely disable mosaic
                    degrees=0.1, translate=0.1, scale=0.0, shear=0.0, 
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=0.0, mixup=0.0)

  trainer = DetectionTrainer(overrides=train_args)
  trainer.add_callback("on_val_end", save_eval_json_with_id)
  trainer.train()

