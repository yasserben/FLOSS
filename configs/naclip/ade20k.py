_base_ = "./base_config.py"

# model settings
model = dict(name_path="./configs/naclip/cls_ade20k.txt", dataset="ade20k")

# dataset settings
dataset_type = "ADE20KDataset"
data_root = "data/ade20k/"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images/training", seg_map_path="annotations/training"
        ),
        pipeline=test_pipeline,
    ),
)


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images/validation", seg_map_path="annotations/validation"
        ),
        pipeline=test_pipeline,
    ),
)
