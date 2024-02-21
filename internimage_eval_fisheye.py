import requests, argparse, warnings, mmcv, torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
# TODO: make more specific imports instead of loading everything
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403
from pathlib import Path

def single_gpu_4batch_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
  model.eval()
  results = []
  dataset = data_loader.dataset
  PALETTE = getattr(dataset, 'PALETTE', None)
  prog_bar = mmcv.ProgressBar(128)
  data_iterator = iter(data_loader)
  for i in range(4):
    data = next(data_iterator)
    with torch.no_grad(): 
      result = model(return_loss=False, rescale=True, **data)
    results.extend(result)

    # The mask head is removed so there's no need to encode mask results
    for _ in range(len(result)): 
      prog_bar.update()

  return results

def det2json(dataset, results):
  json_results = []
  for idx in range(128):
    img_id = dataset.img_ids[idx]
    #filename = dataset.coco.load_imgs(img_id)[0]['file_name']
    result = results[idx]
    for label in range(len(result)):
      if label in [0,2,3,5,7]:
        bboxes = result[label]
        for i in range(bboxes.shape[0]):
          data = dict()
          data['image_id'] = img_id
          # TODO: convert the bbox into normalized coordinates
          data['bbox'] = dataset.xyxy2xywh(bboxes[i])
          data['score'] = float(bboxes[i][4])
          data['category_id'] = label
          json_results.append(data)
  return json_results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    args = parser.parse_args()

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

if __name__ == '__main__':
  # TODO: need a better way to handle different configs, this api is awful
  args = parse_args()
  cfg = Config.fromfile(args.config)
  torch.backends.cudnn.benchmark = True
  cfg.model.pretrained = None
  cfg.data.test.test_mode = True

  # build the dataloader
  dataset = build_dataset(cfg.data.test)
  data_loader = build_dataloader(dataset,
                                 samples_per_gpu=32,
                                 workers_per_gpu=4,
                                 dist=False,
                                 shuffle=False)

  # build the model and load weights
  checkpoint = "checkpoints/cascade_internimage_xl_fpn_3x_coco.pth"
  if not Path(checkpoint).exists():
    print("Model checkpoint doesn't exist. Downloading...")
    response = requests.get("https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth")
    Path(checkpoint).write_bytes(response.content)

  # initialize the pre-trained model
  cfg.model.train_cfg = None
  model = build_detector(cfg.model) # cfg.model contains the threshold for nms as well as score threshold
  checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
  model = fuse_conv_bn(model)
  model.CLASSES = dataset.CLASSES
  model = MMDataParallel(model, device_ids=[0])

  # predict bounding boxes and save results to json
  outputs = single_gpu_4batch_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3)
  results_json = det2json(dataset, outputs)
  mmcv.dump(results_json, "results/internimage_eval.bbox.json")
