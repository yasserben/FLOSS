_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/naclip/cls_context59.txt',
    dataset = "pascalco59"
)

# dataset settings
dataset_type = "PascalContextDataset59"
data_root = "data/VOCdevkit/"

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
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
            img_path="VOC2010/JPEGImages",
            seg_map_path="VOC2010/SegmentationClassContext",
        ),
        ann_file="VOC2010/ImageSets/SegmentationContext/train.txt",
        pipeline=test_pipeline,
    ),
)


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='VOC2010/JPEGImages', seg_map_path='VOC2010/SegmentationClassContext'),
        ann_file='VOC2010/ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))