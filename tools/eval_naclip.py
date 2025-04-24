import os
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.datasets import (
    custom_datasets_naclip,
)  # This and the following should be loaded here because of mmseg module registration
from mmseg.models.segmentors import naclip

# Mapping of dataset types to their number of classes
DATASET_NUM_CLASSES = {
    "CityscapesDataset": 19,
    "PascalContextDataset59": 59,
    "ADE20KDataset": 150,
    "PascalVOCDataset20": 20,
    "COCOStuffDataset": 171,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation with MMSeg")
    parser.add_argument("--config", default="configs/naclip/cityscapes.py")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--attn", default="")
    parser.add_argument("--std", default="")
    parser.add_argument("--template-id", default="")
    parser.add_argument("--pamr", default="")
    parser.add_argument("--save-miou", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        help="dataset to use (e.g., cityscapes, pascalco59, pascalvoc20, ade20k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train"],
        default="val",
        help="dataset split to use for validation/testing (val or train)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "fusion", "compute_metric"],
        default="default",
        help="mode to run the model in",
    )
    parser.add_argument("--work_dir", help="work directory")

    parser.add_argument(
        "--show-dir", default="", help="directory to save visualization images"
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=0,
        help="start index of the template id",
    )
    parser.add_argument(
        "--id-end",
        type=int,
        default=79,
        help="end index of the template id",
    )
    args = parser.parse_args()
    return args


def update_dataset_config(
    cfg, dataset_name: str = None, split: str = "val", mode: str = "default"
) -> None:
    """
    Update configuration with dataset and mode settings.

    Args:
        cfg: The config to update
        dataset_name: Name of the dataset to use (optional)
        split: Which split to use for validation/testing ('val' or 'train')
        mode: Mode to run the model in ('default', 'fusion', or 'compute_metric')
    """
    from mmengine.config import Config
    import os.path as osp

    class_names = None

    # Handle dataset configuration if dataset_name is provided
    if dataset_name:
        if dataset_name.startswith("acdc_"):
            condition = dataset_name.split("_")[1]
            ds_file = osp.join("configs", "naclip", f"{condition}-acdc.py")
        elif dataset_name in [
            "bdd",
            "mapillary",
            "cityscapes",
            "pascalco59",
            "pascalvoc20",
            "ade20k",
            "cocostuff",
        ]:
            ds_file = osp.join("configs", "naclip", f"{dataset_name}.py")
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

        ds_cfg = Config.fromfile(ds_file)

        # List the dataset-related keys to update
        dataset_keys = [
            "model",
            "test_dataloader",
            "train_dataloader",
            "test_pipeline",
            "data_root",
            "dataset_type",
        ]
        # Override the corresponding keys in the main config
        for key in dataset_keys:
            if key in ds_cfg:
                cfg[key] = ds_cfg[key]

    # If split is 'train', replace test dataloader with train dataloader
    if split == "train":
        train_dataloader = cfg.get("train_dataloader")
        if train_dataloader is not None:
            cfg["test_dataloader"]["dataset"] = train_dataloader["dataset"]

    # Handle mode configuration
    if mode != "default":
        # Update model mode
        if "model" not in cfg:
            cfg.model = {}
        cfg.model["mode"] = mode

        # For compute_metric mode, update evaluator settings
        if mode == "compute_metric":
            # Change evaluator type to EntropyMetric
            if "test_evaluator" in cfg:
                cfg.test_evaluator["type"] = "EntropyMetric"
                cfg.test_evaluator["model_name"] = "naclip"

                # Set num_classes based on dataset type
                num_classes = DATASET_NUM_CLASSES.get(cfg.dataset_type)
                if num_classes is not None:
                    cfg.test_evaluator["num_classes"] = num_classes
                else:
                    raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")

            # Update dataset name in evaluator if provided
            if dataset_name:
                cfg.test_evaluator["dataset_name"] = dataset_name
            if cfg.model["type"] == "NACLIP":
                cfg.test_evaluator["model_name"] = "naclip"


def visualization_hook(cfg, show_dir):
    if show_dir == "":
        cfg.default_hooks.pop("visualization", None)
        return
    if "visualization" not in cfg.default_hooks:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks, see base_config.py"
        )
    else:
        hook = cfg.default_hooks["visualization"]
        hook["draw"] = True
        visualizer = cfg.visualizer
        visualizer["save_dir"] = show_dir
        cfg.model["pamr_steps"] = 50
        cfg.model["pamr_stride"] = [1, 2, 4, 8, 12, 24]


def safe_set_arg(cfg, arg, name, func=lambda x: x):
    if arg != "":
        cfg.model[name] = func(arg)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    safe_set_arg(cfg, args.backbone, "clip_path")
    safe_set_arg(cfg, args.arch, "arch")
    safe_set_arg(cfg, args.attn, "attn_strategy")
    safe_set_arg(cfg, args.template_id, "template_id")
    safe_set_arg(cfg, args.std, "gaussian_std", float)
    # PAMR is off
    cfg.model["pamr_steps"] = 0
    visualization_hook(cfg, args.show_dir)

    # Update dataset configuration regardless of whether --dataset is provided
    update_dataset_config(cfg, args.dataset, args.split, args.mode)

    if args.work_dir is not None:
        cfg.work_dir = osp.join("./work_dirs", args.work_dir)
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", "naclip_" + osp.splitext(osp.basename(args.config))[0]
        )

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.save_miou:
        if isinstance(cfg.test_evaluator, list):
            for evaluator in cfg.test_evaluator:
                if evaluator["type"] == "IoUMetric":
                    evaluator["miou_dir"] = cfg.work_dir
        else:
            cfg.test_evaluator["miou_dir"] = cfg.work_dir

    if args.mode == "compute_metric":
        cfg.test_evaluator["id_start"] = args.id_start
        cfg.test_evaluator["id_end"] = args.id_end
        cfg.model["id_start"] = args.id_start
        cfg.model["id_end"] = args.id_end

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
