"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import torch, json, wandb, contextlib, argparse
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from utils import get_image_id
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.nn.modules.conv import autopad
from ultralytics.data.augment import Albumentations
from ultralytics.utils.torch_utils import make_divisible
from ultralytics.models.yolo.detect.train import DetectionTrainer

# TODO: maybe try some different module
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    RepConv,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
    WorldDetect,
)

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

# monkey patching to bypass the unwanted code without modifying the library
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

def load_model_custom(self, cfg=None, weights=None, verbose=True):
  """Return a YOLO detection model."""
  weights, _ = attempt_load_one_weight("checkpoints/yolov8x.pt") 
  model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  # print model state dictionaries
  for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

  if weights:
    model.load(weights)
  return model

class DeformConv(nn.Module):
  from torchvision.ops import deform_conv2d

  default_act = nn.SiLU()  # default activation

  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
  
    #self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    print(self.conv)
    self.bn = nn.BatchNorm2d(c2)
    self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

  def forward(self, x):
    print(f"forward {x.shape}")
    return self.act(self.bn(self.conv(x)))

  def forward_fuse(self, x):
    print(f"forward_fuse {x.shape}")
    return self.act(self.conv(x))

def parse_dcn_model(d, ch, verbose=True):  # model_dict, input_channels(3)
  """Parse a YOLO model.yaml dictionary into a PyTorch model."""
  import ast

  max_channels = float("inf")
  nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
  depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
  if scales:
    scale = d.get("scale")
    if not scale:
      scale = tuple(scales.keys())[0]
      LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
    depth, width, max_channels = scales[scale]

  if act:
    Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    if verbose:
      LOGGER.info(f"{colorstr('activation:')} {act}")  # print

  if verbose:
    LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
  ch = [ch]
  layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
  for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
    m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
    for j, a in enumerate(args):
      if isinstance(a, str):
        with contextlib.suppress(ValueError):
          args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

    n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
    if m in (
      Classify,
      Conv,
      ConvTranspose,
      GhostConv,
      Bottleneck,
      GhostBottleneck,
      SPP,
      SPPF,
      DWConv,
      Focus,
      BottleneckCSP,
      C1,
      C2,
      C2f,
      C2fAttn,
      C3,
      C3TR,
      C3Ghost,
      nn.ConvTranspose2d,
      DWConvTranspose2d,
	  DeformConv,
      C3x,
      RepC3,
    ):
      c1, c2 = ch[f], args[0]
      if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
        c2 = make_divisible(min(c2, max_channels) * width, 8)
      if m is C2fAttn:
        args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
        args[2] = int(
            max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
        )  # num heads

      args = [c1, c2, *args[1:]]
      if m in (BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3):
        args.insert(2, n)  # number of repeats
        n = 1
    elif m is AIFI:
      args = [ch[f], *args]
    elif m in (HGStem, HGBlock):
      c1, cm, c2 = ch[f], args[0], args[1]
      args = [c1, cm, c2, *args[2:]]
      if m is HGBlock:
        args.insert(4, n)  # number of repeats
        n = 1
    elif m is ResNetLayer:
      c2 = args[1] if args[3] else args[1] * 4
    elif m is nn.BatchNorm2d:
      args = [ch[f]]
    elif m is Concat:
      c2 = sum(ch[x] for x in f)
    elif m in (Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn):
      args.append([ch[x] for x in f])
      if m is Segment:
        args[2] = make_divisible(min(args[2], max_channels) * width, 8)
    elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
      args.insert(1, [ch[x] for x in f])
    else:
      c2 = ch[f]

    m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
    t = str(m)[8:-2].replace("__main__.", "")  # module type
    m.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
    if verbose:
      LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
    save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    layers.append(m_)
    if i == 0:
      ch = []
    ch.append(c2)
  return nn.Sequential(*layers), sorted(save)

if __name__ == "__main__":

  # monkey patches
  Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int,   default=1,   help="batch size")
  parser.add_argument('-frac',    type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch',   type=int,   default=1,   help="number of epoch")
  parser.add_argument('-bs',      type=int,   default=16,  help="number of batches")
  parser.add_argument('-project', type=str,   default="fisheye-challenge", help="project name")
  parser.add_argument('-name',    type=str,   default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(project=args.project, name=args.name,
                    model="yolov8x_dcn.yaml", data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, fraction=args.frac, imgsz=1280,
                    exist_ok=True,
                    val=True, save_json=True, conf=0.001, iou=0.7,
                    optimizer="Adam", seed=0,
                    box=7.5, cls=0.125, dfl=3.0,
                    close_mosaic=0,
                    degrees=0.1, translate=0.1, scale=0.0, shear=0.0, 
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=0.0, mixup=0.0)

  trainer = DetectionTrainer(overrides=train_args)
  trainer.add_callback("on_val_end", save_eval_json_with_id)
  trainer.train()

