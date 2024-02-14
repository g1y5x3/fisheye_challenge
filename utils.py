import wandb
from PIL import Image

def bounding_boxes(img, boxes_predict, boxes_gt, class_coco, classid_fisheye):
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
                   "box_caption": class_coco[int(classid_fisheye[int(box.split()[0])])],
                 }
                 for box in boxes_gt
               ],
               "class_labels": class_coco,
             },
             "prediction": {
               "box_data": [
                 {
                   "position": {
                     "minX": float(box.xyxyn.cpu().numpy()[0][0]),
                     "minY": float(box.xyxyn.cpu().numpy()[0][1]),
                     "maxX": float(box.xyxyn.cpu().numpy()[0][2]),
                     "maxY": float(box.xyxyn.cpu().numpy()[0][3]),
                   },
                   "class_id"   : int(box.cls.cpu().numpy()[0]),
                   "box_caption": class_coco[int(box.cls.cpu().numpy()[0])],
                   "scores"     : {"score": float(box.conf.cpu().numpy()[0])}
                 }
                 for box in boxes_predict
               ],
               "class_labels": class_coco,
             }
           },
        ) 
