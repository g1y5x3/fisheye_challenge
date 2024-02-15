import torch, wandb
from PIL import Image
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.models.yolo.detect.val import DetectionValidator

class FisheyeDetectionValidator(DetectionValidator):
  # modifiedy from
  # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py
  # to make it more framework agnostic
  # preds [x, y, x, y, conf, cls]: list of predictions from each image
  # gts   [x, y, x, y, cls]      : list of ground truths from the same image
  # NOTE: currently x, y, w, and h are provided as normalized coordinate

  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.training = False

  def init_metrics(self, names):
    self.names = names
    self.nc = len(names)
    self.metrics.names = self.names
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=0.25, iou_thres=0.5, task="detect")
    self.seen = 0
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

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

def bounding_boxes(img, boxes_predict, boxes_gt, class_name, classid_coco, classid_fisheye):
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
