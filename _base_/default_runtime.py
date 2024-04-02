checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'fisheye-challenge', 'entity': 'g1y5x3', 'name': 'internimage-baseline'},
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
