import wandb
import numpy as np
from PIL import Image
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator, Colors

def plot_images(images, cls, bboxes, confs, names):
  bs = 1
  colors = Colors()
  ns = np.ceil(bs**0.5)
  h, w, _ = images.shape 
  mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
  i = 0
  x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
  mosaic[y : y + h, x : x + w, :] = images
  
  # Resize
  max_size = 1920
  scale = max_size / ns / max(h, w)
  
  # Annotate
  fs = int((h + w) * ns * 0.005)  # font size
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
          if labels or conf[j] > 0.25:  # 0.25 conf thresh
            label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
            annotator.box_label(box, label, color=color, rotated=None)
  annotator.show()

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
