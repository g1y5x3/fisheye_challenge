import torch
import wandb
import numpy as np
from PIL import Image
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator, Colors
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.models.yolo.detect.val import DetectionValidator

# modifiedy from ultralytics to make it more framework agnostic, the original function was too tie
# to batch processing workflow
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py
# 
# preds [[x, y, x, y, conf, cls], ...]: list of torch.tensors [N, 6]
# gts   [[x, y, x, y, cls], ...]      : list of torch.tensors [M, 5]
#
class FisheyeDetectionValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.training = False

  def init_metrics(self, names):
    self.names = names
    self.nc = len(names)
    self.metrics.names = self.names
    # default: conf=0.25, iou_thres=0.45
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, task="detect")
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    self.seen = 0

  # TODO: refactor this
  def update_metrics(self, preds, gts):
    for pred, gt in zip(preds, gts):
      self.seen += 1
      npr = len(pred)
      stat = dict(
        conf=torch.zeros(0),
        pred_cls=torch.zeros(0),
        tp=torch.zeros(npr, self.niou, dtype=torch.bool),
      )
      bbox, cls = gt[:, :4], gt[:,4]
      nl = len(cls)
      stat["target_cls"] = cls
      if npr == 0:
        if nl:
          for k in self.stats.keys():
            self.stats[k].append(stat[k])
          self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
      else:
        stat["conf"], stat["pred_cls"] = pred[:, 4], pred[:, 5]
        if nl:
          stat["tp"] = self._process_batch(pred, bbox, cls)
          self.confusion_matrix.process_batch(detections=pred, gt_bboxes=bbox, gt_cls=cls)
        for k in self.stats.keys():
          self.stats[k].append(stat[k])

# modified from 
# 
def plot_images(images, cls, bboxes, confs, names, title=None, conf_thres=0.25, plot=False):
  # TODO: plot multiple images
  bs = 1
  colors = Colors()
  ns = np.ceil(bs**0.5)
  h, w, _ = images.shape 
  # Not all images have the same dimension, need to resize then before reshaping into a single
  # collection
  mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
  i = 0
  x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
  mosaic[y : y + h, x : x + w, :] = images
  
  # Resize
  max_size = 1920
  scale = max_size / ns / max(h, w)
  
  # Annotate
  fs = int((h + w) * ns * 0.01)  # font size
  annotator = Annotator(images, line_width=round(fs / 10), font_size=fs, pil=True, example=None)
  for i in range(bs):
    x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
    annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
    if len(cls) > 0:
      classes = cls.astype("int")
      labels = confs is None
      
      if len(bboxes):
        boxes = bboxes
        conf = confs if confs is not None else None  # check for confidence presence (label vs pred)
        boxes = ops.ltwh2xyxy(boxes)
        boxes[..., 0::2] += x
        boxes[..., 1::2] += y
        for j, box in enumerate(boxes.astype(np.int64).tolist()):
          c = classes[j]
          color = colors(c)
          c = names[c] if names else c
          if labels or conf[j] > conf_thres:
            label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
            annotator.box_label(box, label, color=color, rotated=None)

  if plot: annotator.show()
  
  return annotator.result()

def bounding_boxes(img, boxes_predict, boxes_gt, class_name):
  
  return wandb.Image(
           img,
           boxes={
             "groundtruth": {
               "box_data": [
                 {
                   "position": {
                       "middle": [float(box.split()[1]), float(box.split()[2])],
                       "width" : float(box.split()[3]),
                       "height": float(box.split()[4]),
                   },
                   "class_id": int(classid_fisheye[int(box.split()[0])]),
                   "box_caption": class_name[int(classid_fisheye[int(box.split()[0])])],
                 }
                 for box in boxes_gt
               ],
               "class_labels": class_name,
             },
             "prediction": {
               "box_data": [
                 {
                   "position": {
                     "minX": float(box.xyxyn.cpu().numpy()[0][0]),  # check how to avoid the first indexing
                     "minY": float(box.xyxyn.cpu().numpy()[0][1]),
                     "maxX": float(box.xyxyn.cpu().numpy()[0][2]),
                     "maxY": float(box.xyxyn.cpu().numpy()[0][3]),
                   },
                   "class_id"   : int(classid_coco[int(box.cls.cpu().numpy()[0])]),
                   "box_caption": class_name[int(classid_coco[int(box.cls.cpu().numpy()[0])])],
                   "scores"     : {"score": float(box.conf.cpu().numpy()[0])}
                 }
                 for box in boxes_predict
               ],
               "class_labels": class_name,
             }
           },
        ) 
