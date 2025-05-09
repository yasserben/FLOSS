# Model settings
model = dict(
    type="DinoCLIP_Inferencer",
    model=dict(
        type="MaskClip",
        class_names=[
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
        decode_head=dict(
            type="MaskClipHead",
            in_channels=768,
            text_channels=512,
            align_corners=False,
            use_templates=True,
            pretrained="laion2b_s34b_b88k",
            fuse_predictions=False,
        ),
        clip_model="ViT-B-16",
        backbone=dict(
            img_size=224,
            patch_size=16,
        ),
    ),
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        bgr_to_rgb=True,
    ),
    num_classes=19,
    test_cfg=dict(mode="slide", stride=(224, 224), crop_size=(448, 448)),
    model_name="maskclip",
)
