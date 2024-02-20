# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import argparse
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403

# just to reduce the number of batches for quick testing
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

    batch_size = len(result)

    # The mask head is removed so there's no need to encode mask results
    results.extend(result)

    for _ in range(batch_size):
      prog_bar.update()

  return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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


def main():
  args = parse_args()

  cfg = Config.fromfile(args.config)
  if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
  # set cudnn_benchmark
  if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

  cfg.model.pretrained = None
  cfg.data.test.test_mode = True
  cfg.gpu_ids = range(1)

  # build the dataloader
  print(cfg.data.test)
  dataset = build_dataset(cfg.data.test)
  data_loader = build_dataloader(dataset,
                                 samples_per_gpu=32,
                                 workers_per_gpu=4,
                                 dist=False,
                                 shuffle=False)

  # build the model and load checkpoint
  cfg.model.train_cfg = None
  model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
  checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
  model = fuse_conv_bn(model)
  model.CLASSES = dataset.CLASSES
  model = MMDataParallel(model, device_ids=cfg.gpu_ids)

  # predict bounding boxes
  outputs = single_gpu_4batch_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3)
  mmcv.dump(outputs, "results/internimage_eval.pkl")

  #TODO: Retrieve the image id with the test image sequences and upload to wandb using val.py

  #kwargs = {} if args.eval_options is None else args.eval_options
  #if args.format_only:
  #    dataset.format_results(outputs, **kwargs)
  #if args.eval:
  #    eval_kwargs = cfg.get('evaluation', {}).copy()
  #    # hard-code way to remove EvalHook args
  #    for key in [
  #            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
  #            'rule', 'dynamic_intervals'
  #    ]:
  #        eval_kwargs.pop(key, None)
  #    eval_kwargs.update(dict(metric=args.eval, **kwargs))
  #    metric = dataset.evaluate(outputs, **eval_kwargs)
  #    print(metric)
  #    metric_dict = dict(config=args.config, metric=metric)
  #    if args.work_dir is not None and rank == 0:
  #        mmcv.dump(metric_dict, json_file)

if __name__ == '__main__':
    main()
