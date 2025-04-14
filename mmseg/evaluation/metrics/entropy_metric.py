import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
import json
import os
from pathlib import Path
from mmseg.models.utils import resize
import fcntl  # For file locking
import time
from mmseg.registry import METRICS


@METRICS.register_module()
class EntropyMetric(BaseMetric):
    """Entropy evaluation metric for semantic segmentation.

    Args:
        num_classes (int): Number of classes in the dataset.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        model_name (str, optional): Name of the model for analysis.
            Defaults to None.
        dataset_name (str, optional): Name of the dataset for analysis.
            Defaults to None.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(
        self,
        num_classes: int = 19,
        num_templates: int = 80,
        collect_device: str = "cpu",
        # original_size: Tuple[int, int] = (1024, 2048),
        output_dir: Optional[str] = None,
        model_name: str = "clipdinoiser",
        dataset_name: str = "cityscapes",
        prefix: Optional[str] = None,
        id_start: int = 0,
        id_end: int = 79,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        # self.original_size = original_size
        self.num_templates = num_templates
        self.id_start = id_start
        self.id_end = id_end
        self.process_all_templates = (
            self.id_end - self.id_start + 1
        ) == self.num_templates

        if is_main_process():
            # Initialize entropy_data only for the specified template range
            self.entropy_data = {
                str(t): {
                    str(i): {"sum": np.float64(0.0), "count": np.float64(0)}
                    for i in range(self.num_classes)
                }
                for t in range(self.id_start, self.id_end + 1)
            }
            self.total_pixels = np.float64(0)
            self.file_count = 0

    def process(self, data_batch: dict, data_samples: Sequence[torch.Tensor]) -> None:
        """Process one batch of data and data_samples.

        Args:
            data_batch (dict): A batch of data from the dataloader
            data_samples (Sequence[torch.Tensor]): A batch of probability tensors
                from the model for templates in the specified range.
        """
        if not is_main_process():
            return

        # Get metadata from data_batch
        img_meta = data_batch["data_samples"][0].metainfo

        # Get original dimensions to update total_pixels
        H, W = img_meta["ori_shape"][:2]
        self.total_pixels += np.float64(H * W)
        self.file_count += 1

        # Process each template in the specified range
        for idx, prob in enumerate(data_samples):
            template_idx = idx + self.id_start

            # Skip if outside our range
            if template_idx > self.id_end:
                continue

            template_key = str(template_idx)

            # Skip if this template isn't in our tracking dict
            if template_key not in self.entropy_data:
                continue

            if self.model_name != "naclip":
                # Process the template
                prob = prob.unsqueeze(0)
                prob = resize(
                    prob,
                    size=img_meta["ori_shape"],
                    mode="bilinear",
                    align_corners=False,
                    warning=False,
                )
                prob = prob.squeeze(0)
            pred_label = torch.argmax(prob, dim=0)

            # Compute entropy for each class
            for class_idx in range(self.num_classes):
                class_mask = pred_label == class_idx
                if not torch.any(class_mask):
                    continue

                class_probs = prob[:, class_mask].to(torch.float32)
                epsilon = 1e-10
                log_probs = torch.log(class_probs + epsilon)
                entropy = -(class_probs * log_probs).sum(dim=0)
                mean_entropy = entropy.mean().item()
                count = class_mask.sum().item()

                class_key = str(class_idx)
                self.entropy_data[template_key][class_key]["sum"] += np.float64(
                    mean_entropy * count
                )
                self.entropy_data[template_key][class_key]["count"] += np.float64(count)

            del prob
            torch.cuda.empty_cache()

    def acquire_lock(
        self, file_path: Path, max_attempts: int = 60, wait_time: float = 1.0
    ) -> bool:
        """Attempt to acquire a file lock with timeout."""
        lock_path = file_path.parent / f"{file_path.name}.lock"
        attempts = 0

        while attempts < max_attempts:
            try:
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (IOError, OSError):
                time.sleep(wait_time)
                attempts += 1

        return False

    def release_lock(self, file_path: Path) -> None:
        """Release the file lock."""
        lock_path = file_path.parent / f"{file_path.name}.lock"
        try:
            os.remove(lock_path)
        except OSError:
            pass

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()
        metrics = {}

        if not is_main_process():
            return metrics

        class_names = self.dataset_meta["classes"]
        if self.model_name == "naclip":
            class_names = [a.replace(" ", "") for a in class_names]

        analysis_dir = Path(f"./rankings/{self.model_name}")
        mkdir_or_exist(analysis_dir)
        json_file_path = (
            analysis_dir
            / f"template_rankings_{self.model_name}_{self.dataset_name}.json"
        )

        # Try to acquire lock
        if not self.acquire_lock(json_file_path):
            print_log("Failed to acquire lock for writing rankings", logger=logger)
            return metrics

        try:
            # Load existing rankings or create new ones
            if json_file_path.exists():
                with open(json_file_path, "r") as f:
                    rankings = json.load(f)
            else:
                rankings = {
                    "model": self.model_name,
                    "dataset": self.dataset_name,
                    "split": "train",
                    "classes": {},
                }

            # Process each class
            for class_idx, class_name in enumerate(class_names):
                if class_name not in rankings["classes"]:
                    rankings["classes"][class_name] = {}

                class_rankings = rankings["classes"][class_name]
                templates_data = []

                # Process templates in our range
                for template_idx in range(self.id_start, self.id_end + 1):
                    template_key = str(template_idx)
                    class_key = str(class_idx)
                    class_data = self.entropy_data[template_key][class_key]

                    if class_data["count"] > 0:
                        entropy_value = float(class_data["sum"]) / float(
                            class_data["count"]
                        )
                        pixel_percentage = (
                            float(class_data["count"]) / float(self.total_pixels) * 100
                        )
                    else:
                        entropy_value = 0.0
                        pixel_percentage = 0.0

                    template_data = {
                        "template_id": template_idx,
                        "entropy": entropy_value,
                        "pixel_percentage": pixel_percentage,
                    }
                    templates_data.append(template_data)

                # Update existing rankings or create new ones
                existing_rankings = class_rankings.get("entropy_ranking", [])

                # Create a map of existing template data
                existing_template_map = {
                    str(t["template_id"]): t for t in existing_rankings
                }

                # Update or add new template data
                for template_data in templates_data:
                    template_id = str(template_data["template_id"])
                    if template_id in existing_template_map:
                        existing_template_map[template_id].update(template_data)
                    else:
                        existing_template_map[template_id] = template_data

                # Convert back to list and sort by entropy
                all_templates = list(existing_template_map.values())
                entropy_sorted = sorted(all_templates, key=lambda x: x["entropy"])

                # Update rankings
                class_rankings["entropy_ranking"] = [
                    {**t, "rank": i + 1} for i, t in enumerate(entropy_sorted)
                ]

            # Write updated rankings
            with open(json_file_path, "w") as f:
                json.dump(rankings, f, indent=2)

            print_log(
                f"Updated template rankings saved to {json_file_path}", logger=logger
            )

        finally:
            # Always release the lock
            self.release_lock(json_file_path)

        return metrics
