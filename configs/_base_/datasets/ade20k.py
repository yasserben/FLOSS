# Remove the imports and with block
dataset_type = "ADE20KDataset"
dataset_config = dict(
    root="data/ade20k/",
    train_scale=(2048, 448),
    test_scale=(2048, 448),
    crop_size=(448, 448),
)

# Configuration parameters
cfg_params = dict(
    validate_on_train=False,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=dataset_config["test_scale"], keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs", divide_by_255=True),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=dataset_config["test_scale"], keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs", divide_by_255=True),
]

train_cityscapes = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path="images/training",
        seg_map_path="annotations/training",
    ),
    pipeline=train_pipeline,
)

val_cityscapes = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path="images/validation",
        seg_map_path="annotations/validation",
    ),
    pipeline=test_pipeline,
)

# DataLoader settings
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_cityscapes,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_cityscapes,
)

test_dataloader = val_dataloader

# Evaluator settings
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"], miou_dir=None)
test_evaluator = val_evaluator
