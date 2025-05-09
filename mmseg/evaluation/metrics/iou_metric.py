# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


def colorize_mask(mask):
    """Colorize segmentation mask using Cityscapes palette."""
    palette = [
        128,
        64,
        128,  # road
        244,
        35,
        232,  # sidewalk
        70,
        70,
        70,  # building
        102,
        102,
        156,  # wall
        190,
        153,
        153,  # fence
        153,
        153,
        153,  # pole
        250,
        170,
        30,  # traffic light
        220,
        220,
        0,  # traffic sign
        107,
        142,
        35,  # vegetation
        152,
        251,
        152,  # terrain
        70,
        130,
        180,  # sky
        220,
        20,
        60,  # person
        255,
        0,
        0,  # rider
        0,
        0,
        142,  # car
        0,
        0,
        70,  # truck
        0,
        60,
        100,  # bus
        0,
        80,
        100,  # train
        0,
        0,
        230,  # motorcycle
        119,
        11,
        32,  # bicycle
    ]

    # Pad palette to 256*3
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(palette)
    return new_mask


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        iou_metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        collect_device: str = "cpu",
        output_dir: Optional[str] = None,
        format_only: bool = False,
        prefix: Optional[str] = None,
        save_miou: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.save_miou = save_miou
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta["classes"])

        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                self.results.append(
                    self.intersect_and_union(
                        pred_label, label, num_classes, self.ignore_index
                    )
                )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f"results are saved to {osp.dirname(self.output_dir)}")
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta["classes"]

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)

        # Save results to JSON file
        if self.save_miou is not None:
            if not osp.exists(self.save_miou):
                mkdir_or_exist(self.save_miou)

            # Create dictionary with class-wise IoUs and mean IoU
            miou_results = {
                "per_class_iou": {
                    class_name: float(
                        f"{iou:.2f}"
                    )  # Convert to float for JSON serialization
                    for class_name, iou in zip(
                        ret_metrics_class["Class"], ret_metrics_class["IoU"]
                    )
                },
                "mean_iou": float(f"{np.mean(ret_metrics_class['IoU']):.2f}"),
            }

            json_file_path = osp.join(self.save_miou, "mIoU_results.json")
            with open(json_file_path, "w") as f:
                json.dump(miou_results, f, indent=4)

            print_log(f"mIoU results saved to {json_file_path}", logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(
        pred_label: torch.tensor,
        label: torch.tensor,
        num_classes: int,
        ignore_index: int,
    ):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
        ).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
        ).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0, max=num_classes - 1
        ).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (
                (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f"metrics {metrics} is not supported")

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics
