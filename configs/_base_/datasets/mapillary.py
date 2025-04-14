dataset_type = "CityscapesDataset"
dataset_config = dict(
    root="data/mapillary/",
    train_scale=(1920, 448),
    test_scale=(1920, 448),
    crop_size=(448, 448),
    img_suffix=".jpg",
    seg_map_suffix=".png",
)

# Configuration parameters
cfg_params = dict(
    use_tube=False,  # Override default tube setting for BDD
    validate_on_train=False,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=dataset_config["train_scale"]),
    dict(type="RandomCrop", crop_size=dataset_config["crop_size"]),
    dict(type="PackSegInputs", divide_by_255=True),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=dataset_config["test_scale"], keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs", divide_by_255=True),
]

train_bdd = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path="training/v1.2/images",
        seg_map_path="cityscapes_trainIdLabel/train/label",
    ),
    img_suffix=dataset_config["img_suffix"],
    seg_map_suffix=dataset_config["seg_map_suffix"],
    pipeline=train_pipeline,
)

val_bdd = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path="half/val_img",
        seg_map_path="half/val_label",
    ),
    img_suffix=dataset_config["img_suffix"],
    seg_map_suffix=dataset_config["seg_map_suffix"],
    pipeline=test_pipeline,
)

# DataLoader settings
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_bdd,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_bdd,
)

test_dataloader = val_dataloader

# Evaluator settings
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"], miou_dir=None)
test_evaluator = val_evaluator
