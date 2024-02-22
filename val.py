import torch
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.models.yolo.detect.val import DetectionValidator

# modifiedy from ultralytics to make it more framework agnostic
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
