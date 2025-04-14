# Training settings
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00001, weight_decay=0.0005),
    # clip_grad=dict(max_norm=1, norm_type=2),
)

# Common scheduler settings
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=11250, by_epoch=False)
]

# Common training settings
train_cfg = dict(type="IterBasedTrainLoop", max_iters=5000, val_interval=250)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Common hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=5000, max_keep_ckpts=1
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
# Common model settings
find_unused_parameters = True
