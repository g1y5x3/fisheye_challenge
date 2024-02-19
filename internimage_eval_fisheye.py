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
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--gpu-ids',
                        type=int,
                        nargs='+',
                        help='ids of gpus to use '
                        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results.')
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
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

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

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.data.test.test_mode = True
    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    cfg.gpu_ids = range(1)
    rank, _ = get_dist_info()

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

    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)

    mmcv.dump(outputs, args.out)

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
