dataset_type = "CityscapesDataset"
dataset_config = dict(
    root="data/acdc/",
    train_scale=(1920, 1080),
    test_scale=(796, 448),
    crop_size=(448, 448),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
)

# Configuration parameters
cfg_params = dict(
    use_tube=False,
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

train_night_acdc = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path="rgb_anon/night/train",
        seg_map_path="gt/night/train",
        **(
            dict(tube_path="tube_normalized/night/train")
            if cfg_params["use_tube"]
            else {}
        ),
    ),
    img_suffix=dataset_config["img_suffix"],
    seg_map_suffix=dataset_config["seg_map_suffix"],
    pipeline=train_pipeline,
)

val_night_acdc = dict(
    type=dataset_type,
    data_root=dataset_config["root"],
    data_prefix=dict(
        img_path=(
            "rgb_anon/night/train"
            if cfg_params["validate_on_train"]
            else "rgb_anon/night/val"
        ),
        seg_map_path=(
            "gt/night/train" if cfg_params["validate_on_train"] else "gt/night/val"
        ),
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
    dataset=train_night_acdc,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_night_acdc,
)

test_dataloader = val_dataloader

# Evaluator settings
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"], miou_dir=None)
test_evaluator = val_evaluator
