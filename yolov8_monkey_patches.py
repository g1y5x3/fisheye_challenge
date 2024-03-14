import contextlib
import torch
import torch.nn as nn
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import make_divisible
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
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
from utils import DeformableConv

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
      #A.OpticalDistortion(p=0.01),
    ]
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
  except ImportError:  # package not installed, skip
    pass
  except Exception as e:
    LOGGER.info(f"{prefix}{e}")

def load_model_custom(self, cfg=None, weights=None, verbose=True):
  """Return a YOLO detection model."""
  # TODO: train yolov8x-p2 to get the initial weights
  print(cfg)
  weights, _ = attempt_load_one_weight("checkpoints/yolov8x.pt")
  model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  print(f"config pretrained: {self.args.pretrained}")
  if weights and self.args.pretrained:
    model.load(weights)
  return model

# TODO: fix this
def get_flops_pass(model, imgsz=640):
  return 0.0

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
      DeformableConv,
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


