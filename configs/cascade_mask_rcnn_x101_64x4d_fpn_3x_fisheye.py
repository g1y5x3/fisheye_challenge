# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
  '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
  '../_base_/datasets/fisheye_instance.py',
  '../_base_/schedules/schedule_3x.py',
  '../_base_/default_runtime.py'
]
model = dict(
  backbone=dict(
    type='ResNeXt',
    depth=101,
    groups=64,
    base_width=4,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    style='pytorch',
    init_cfg=dict(
      type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
  dict(type='LoadImageFromFile'),
  dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
  dict(type='RandomFlip', flip_ratio=0.5),
  dict(type='AutoAugment',
    policies=[
      [
        dict(type='Resize',
          img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                     (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                     (736, 1333), (768, 1333), (800, 1333)],
          multiscale_mode='value',
          keep_ratio=True)
      ],
      [
        dict(type='Resize',
          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
          multiscale_mode='value',
          keep_ratio=True),
        dict(type='RandomCrop',
          crop_type='absolute_range',
          crop_size=(384, 600),
          allow_negative_crop=True),
        dict(type='Resize',
          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                     (576, 1333), (608, 1333), (640, 1333),
                     (672, 1333), (704, 1333), (736, 1333),
                     (768, 1333), (800, 1333)],
          multiscale_mode='value',
          override=True,
          keep_ratio=True)
      ]
    ]),
  dict(type='Normalize', **img_norm_cfg),
  dict(type='Pad', size_divisor=32),
  dict(type='DefaultFormatBundle'),
  dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# we use 4 nodes to train this model, with a total batch size of 64
data = dict(samples_per_gpu=2, train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(
  _delete_=True, type='AdamW', lr=0.0001 * 2, weight_decay=0.05,
  constructor='CustomLayerDecayOptimizerConstructor',
  paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.90,
                     depths=[5, 5, 24, 5], offset_lr_scale=0.01)
  )
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
evaluation = dict(metric=['bbox'], save_best='auto')
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
resume_from = None
custom_hooks = [
  dict(
    type='ExpMomentumEMAHook',
    resume_from=resume_from,
    momentum=0.0001,
    priority=49)
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'fisheye-challenge', 'entity': 'g1y5x3', 'name': 'resnet101-rcnn-cascade-baseline'},
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ]
)
