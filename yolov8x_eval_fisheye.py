import io, os, cv2, torch, wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, Metric
from ultralytics.models.yolo.detect.val import DetectionValidator
from utils import bounding_boxes 

WANDB = os.getenv("WANDB", False)

class FisheyeDetectionValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)

  def init_metrics(self, names):
    self.names = names
    self.nc = len(names)
    self.metrics.names = self.names
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=0.25, iou_thres=0.5, task="detect")
    self.seen = 0
    self.jdict= []
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

  # modifiedy from 
  # https://github.com/ultralytics/ultralytics/blob/5e81651b4f85fb3d404148eedc6b0513628514f2/ultralytics/models/yolo/detect/val.py#L117 
  # to make it more framework agnostic
  # preds [x, y, x, y, conf, cls]: list of predictions from each image
  # gts   [x, y, w, h, cls]      : list of ground truths from the same image
  def update_metrics(self, preds, gts):
    for pred, gt in zip(preds, gts):
      self.seen += 1
      npr = len(pred)
      stat = dict(
        # TODO: add device=self.device back later
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
        continue

    stat["conf"], stat["pred_cls"] = pred[:, 4], pred[:, 5]
    if nl:
      print(preds)
      print(bbox)
      print(cls)
      stat["tp"] = self._process_batch(preds, bbox, cls)
      self.confusion_matrix.process_batch(detection=pred, gt_bboxes=bbox, gt_cls=cls)
    for k in self.stats.keys():
      self.stats[k].append(stat[k])
    
      
if __name__ == "__main__":
  if WANDB:
    run = wandb.init(project="fisheye-challenge", name="yolov8x_eval_128")
    table = wandb.Table(columns=["ID", "Image"])
  
  data_dir  = '../dataset/Fisheye8K_all_including_train/test/images/'
  label_dir = '../dataset/Fisheye8K_all_including_train/test/labels/'
  sources = [data_dir+img for img in os.listdir(data_dir)]
  print(f"Total data for inference {len(sources)}")

  # For the convenience of confusion matrix, all labels are convered to 0 ~ 4, probably can write it in an easier way
  class_name = {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'} 
  classid_fisheye = {0:0, 1:1, 2:2, 3:3, 4:4}   # {0: 'bus', 1: 'motorcycle', 2: 'car', 3: 'person', 4: 'truck'} 
  classid_coco    = {0:3, 2:2, 3:1, 5:0, 7:4}   # {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
 
  model = YOLO('yolov8x.pt') # model was trained on COCO dataset
  conf_mat = ConfusionMatrix(5, conf=0.25, iou_thres=0.5, task="detect")
  fisheye_eval = FisheyeDetectionValidator()
  fisheye_eval.init_metrics(class_name) 
 
  #for i in range(len(sources)//128+1):
  for i in range(1):
    # starting and ending indices for each batch
    start = i*128
    end = (i+1)*128 if i <= 20 else -1 

    results = model.predict(sources[start:end], classes=[0, 2, 3, 5, 7], imgsz=640, conf=0.0, iou=0.7, stream=True)

    preds = []
    gts = []
    for result in results:
      img_id = result.path.rsplit('/',1)[-1]
      print(img_id)

      # Load the groundtruth for corresponding image - [x, y, width, height]
      with open(label_dir + img_id.replace(".png", ".txt"), "r") as file:
        boxes_gt_string = file.readlines()

      # Calculate benchmarks
      # NOTE: might be easier to do if read from json instead
      gt_boxes = torch.empty((len(boxes_gt_string),4))
      gt_cls = torch.empty(len(boxes_gt_string))
      for i, box in enumerate(boxes_gt_string):
        gt_cls[i] = classid_fisheye[int(box.split()[0])]
        gt_boxes[i,:] = ops.xywh2xyxy(torch.tensor([float(box.split()[1]), float(box.split()[2]), float(box.split()[3]), float(box.split()[4])]))
      gt = torch.cat((gt_boxes, gt_cls.unsqueeze(1)), dim=1)
      print("groundtruth", gt.shape)
      gts.append(gt)

      # TODO: apparently when you set conf threhold to 0, the total amount of bounding boxes is capped at 300, 
      # most likely the top 300 ones but need to make sure that's the exact criteria
      cls = torch.tensor([classid_coco[i] for i in result.boxes.cls.cpu().numpy()])
      pred = torch.cat((result.boxes.xyxyn.cpu(), result.boxes.conf.cpu().unsqueeze(1), cls.unsqueeze(1)), dim=1)
      print("prediction", pred.shape)
      preds.append(pred)

      conf_mat.process_batch(pred, gt_boxes, gt_cls)
      
      if WANDB:
        box_img = bounding_boxes(result.orig_img, result.boxes, boxes_gt_string, class_name, classid_coco, classid_fisheye)
        table.add_data(img_id, box_img)

    fisheye_eval.update_metrics(preds, gts)

    # compute benchmarks against the groundtruth
  print(conf_mat.matrix)
    
  if WANDB:
    run.log({"Table" : table})
    run.finish()
