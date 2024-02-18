# modifiey from https://github.com/OpenGVLab/InternImage/blob/master/detection/image_demo.py
import requests, mmcv, asyncio, argparse
# F401: indicates that a module imported in the code is not useda
# F403: indicates that a particular import is shadowed by another import
import mmcv_custom   # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
from pathlib import Path
from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)

if __name__ == '__main__':
  data_dir = "../dataset/Fisheye8K_all_including_train/test/images/"
  config = "configs/cascade_internimage_xl_fpn_3x_coco.py"
  checkpoint = "checkpoints/cascade_internimage_xl_fpn_3x_coco.pth"
  output = "result"

  img = "camera1_A_10.png" 
  img_dir = data_dir + img

  if not Path(checkpoint).exists():
    print("Model checkpoint doesn't exist. Downloading...")
    # ideally to use tqdm to show progress
    response = requests.get("https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth")
    Path(checkpoint).write_bytes(response.content)

  # build the model from a config file and a checkpoint file
  model = init_detector(config, checkpoint, device="cuda:0")
  # test a single image
  result = inference_detector(model, img_dir)
  
  mmcv.mkdir_or_exist(output)
  out_file = Path(output) / Path(img)
  # show the results
  model.show_result(
    img_dir,
    result,
    score_thr=0.3,
    show=False,
    bbox_color="coco",
    text_color=(200, 200, 200),
    mask_color="coco",
    out_file=out_file
  )
  print(f"Result is save at {out_file}")
