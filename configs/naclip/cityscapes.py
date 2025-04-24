_base_ = "./base_config.py"

# model settings
model = dict(name_path="./configs/naclip/cls_cityscapes.txt")

# dataset settings
dataset_type = "CityscapesDataset"
data_root = "data/cityscapes/"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 560), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth does not need to do resize data transform
    dict(type="LoadAnnotations"),
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
        data_prefix=dict(img_path="leftImg8bit/train", seg_map_path="gtFine/train"),
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
        data_prefix=dict(img_path="leftImg8bit/val", seg_map_path="gtFine/val"),
        pipeline=test_pipeline,
    ),
)
