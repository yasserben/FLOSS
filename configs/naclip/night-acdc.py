_base_ = "./base_config.py"

# model settings
model = dict(name_path="./configs/naclip/cls_cityscapes.txt")

# dataset settings
dataset_type = "CityscapesDataset"
data_root = "data/acdc/"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 560), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="rgb_anon/night/val", seg_map_path="gt/night/val"),
        pipeline=test_pipeline,
        img_suffix="_rgb_anon.png",
        seg_map_suffix="_gt_labelTrainIds.png",
    ),
)
