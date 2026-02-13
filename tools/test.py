# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import importlib.util

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work_dir",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
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
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--save-miou", action="store_true", help="Save mIoU")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        help="dataset to use (e.g., bdd, cityscapes, acdc_fog, acdc_rain, acdc_snow, acdc_night, acdc_concat)",
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def update_config(
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
    class_names = None

    # Handle dataset configuration
    if dataset_name:
        # Determine the dataset config file to use
        if dataset_name.startswith("acdc_"):
            condition = dataset_name.split("_")[1]
            ds_file = osp.join("configs", "_base_", "datasets", f"{condition}-acdc.py")
        elif dataset_name in [
            "bdd",
            "mapillary",
            "cityscapes",
            "cocostuff",
            "ade20k",
            "pascalco59",
            "pascalvoc20",
        ]:
            ds_file = osp.join("configs", "_base_", "datasets", f"{dataset_name}.py")
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

        ds_cfg = Config.fromfile(ds_file)

        # List the dataset-related keys to update
        dataset_keys = [
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
            "val_evaluator",
            "test_evaluator",
        ]
        # Override the corresponding keys in the main config
        for key in dataset_keys:
            if key in ds_cfg:
                cfg[key] = ds_cfg[key]

        # Try to get class names from the dataset config
        try:
            # Import class names from the dataset-specific file
            module_path = f"configs._base_.datasets.class_names.{dataset_name}_classes"
            from importlib import import_module

            module = import_module(module_path)
            class_names = module.list_of_classes

            # Update model configuration with class names and dataset
            if "model" not in cfg:
                cfg.model = {}
            if "model" not in cfg.model:
                cfg.model["model"] = {}
            cfg.model["model"]["class_names"] = class_names
            cfg.model["dataset"] = dataset_name
            cfg.model["num_classes"] = len(class_names)

        except ImportError:
            print(f"Warning: Could not import class names from {module_path}")

    # If split is 'train', replace val/test dataloaders with train dataloader
    if split == "train":
        train_dataloader = cfg.get("train_dataloader")
        if train_dataloader is not None:
            cfg["val_dataloader"]["dataset"] = train_dataloader["dataset"]
            cfg["test_dataloader"]["dataset"] = train_dataloader["dataset"]

    # Handle mode configuration
    if mode != "default":
        # Update model mode
        if "model" not in cfg:
            cfg.model = {}
        cfg.model["mode"] = mode

        # For compute_metric mode, update evaluator settings if we have class_names
        if mode == "compute_metric":
            cfg.test_evaluator["type"] = "EntropyMetric"

            # Update num_classes only if we have class_names
            if class_names is not None:
                num_classes = len(class_names)
                cfg.test_evaluator["num_classes"] = num_classes
            if dataset_name:
                # Update dataset name
                cfg.test_evaluator["dataset_name"] = dataset_name
            if cfg.model["model_name"] == "maskclip":
                cfg.test_evaluator["model_name"] = "maskclip"
            # Set ignore_index for IoU computation (255 is typically used for unlabeled pixels)
            cfg.test_evaluator["ignore_index"] = 255


def main():
    args = parse_args()

    # Load config normally.
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # Update configuration with dataset and mode settings
    update_config(cfg, args.dataset, args.split, args.mode)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join("./work_dirs", args.work_dir)
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0] + "_" + args.dataset,
        )

    # Handle checkpoint loading for CLIP-DINOiser
    if cfg.model.get("model_name", "").lower() == "clipdinoiser":
        checkpoint_path = "checkpoints/clip_dinoiser/model_weights.pth"
        if os.path.exists(checkpoint_path):
            cfg.load_from = checkpoint_path
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    if args.save_miou:
        if isinstance(cfg.test_evaluator, list):
            for evaluator in cfg.test_evaluator:
                if evaluator["type"] == "IoUMetric":
                    evaluator["save_miou"] = cfg.work_dir
        else:
            cfg.test_evaluator["save_miou"] = cfg.work_dir


    cfg.test_evaluator["id_start"] = args.id_start
    cfg.test_evaluator["id_end"] = args.id_end
    cfg.model["id_start"] = args.id_start
    cfg.model["id_end"] = args.id_end

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == "__main__":
    main()
