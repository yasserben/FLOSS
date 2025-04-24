import os
import time
import json
from enum import Enum
import gc  # Add this at the top with other imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from typing import List, Optional, Dict, Tuple
from my_utils import save_rgb, save_semantic_map, save_semantic_map_maxed
from mmengine.structures import PixelData
import numpy as np
import torchvision.utils as vutils
from mmseg.utils import (
    ConfigType,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
    add_prefix,
)

# from mmseg.ops import resize
# from mmseg.structures import resize
from mmseg.models.utils import resize
from mmengine.model import BaseModel
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.segmentors.base import BaseSegmentor

from omegaconf import OmegaConf
from mmseg.registry import MODELS
from mmseg.models import build_model, build_model_custom

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@MODELS.register_module()
class DinoCLIP_Inferencer(EncoderDecoder):
    def __init__(
        self,
        model,
        num_classes,
        data_preprocessor,
        top_n=4,
        mode="default",
        dataset="cityscapes",
        model_name="clipdinoiser",
        id_start=0,
        id_end=79,
        test_cfg=dict(),
        **kwargs,
    ):

        super(EncoderDecoder, self).__init__(data_preprocessor)
        if model_name == "clipdinoiser":
            model["clip_backbone"]["decode_head"]["mode"] = mode
            model["clip_backbone"]["decode_head"]["id_start"] = id_start
            model["clip_backbone"]["decode_head"]["id_end"] = id_end
        elif model_name == "maskclip":
            model["decode_head"]["mode"] = mode
            model["decode_head"]["id_start"] = id_start
            model["decode_head"]["id_end"] = id_end
        self.model = MODELS.build(model)
        self.mode = mode
        self.num_classes = num_classes
        self.test_cfg = test_cfg
        self.dataset = dataset
        if self.mode == "fusion":
            self.template_dict = self._load_template_rankings(
                top_n,
                dataset,
                model_name,
            )
        else:
            self.template_dict = None
        if self.mode == "compute_metric":
            self.num_templates = id_end - id_start + 1

        # self.init_weights()
        # print("checkpoints loaded")

    def _load_template_rankings(
        self,
        top_n=4,
        dataset="cityscapes",
        model_name="clipdinoiser",
    ):
        """Load and process template rankings based on specified metric and threshold."""
        # Load the json file
        with open(
            f"rankings/{model_name}/template_rankings_{model_name}_{dataset}.json",
            "r",
        ) as f:
            rankings_data = json.load(f)

        # Use hardcoded selection mode
        selection_mode = "entropy"
        ranking_field = f"{selection_mode}_ranking"

        # Create template dict by selecting templates for each class
        template_dict = {}

        for class_name, class_data in rankings_data["classes"].items():
            if ranking_field in class_data:
                # Filter templates based on pixel percentage threshold if specified
                templates = class_data[ranking_field]
                templates = [
                    template
                    for template in templates
                    if template["pixel_percentage"] > 0.0
                ]

                # Extract template IDs from the filtered ranking list
                template_ids = [template["template_id"] for template in templates][
                    :top_n
                ]
                if template_ids:  # Only add if there are valid templates
                    template_dict[class_name] = template_ids
                else:
                    template_dict[class_name] = None

        return template_dict

    def _force_gpu_cleanup(self):
        """Force GPU memory cleanup by emptying cache and synchronizing."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        gc.collect()  # Force Python garbage collection
        torch.cuda.empty_cache()  # Empty cache again after garbage collection

    def expert_fusion(
        self,
        expert_probs: torch.Tensor,  # [num_experts, 1, C, H, W]
        default_prob: torch.Tensor,  # [1, C, H, W]
    ) -> torch.Tensor:
        """
        Merge predictions using expert fusion with highest confidence strategy.
        Works with single image (no batch dimension) for simplicity.

        Args:
            expert_probs: Tensor of shape [num_experts, 1, C, H, W] containing expert probabilities
            default_prob: Tensor of shape [1, C, H, W] containing default model probabilities

        Returns:
            torch.Tensor: Final prediction probabilities of shape [1, C, H, W]
        """
        # Remove batch dimension since we know it's always 1
        expert_probs = expert_probs.squeeze(1)  # [num_experts, C, H, W]
        default_prob = default_prob.squeeze(0)  # [C, H, W]

        num_experts, C, H, W = expert_probs.shape
        device = expert_probs.device

        # Get predictions from probabilities
        predictions = torch.argmax(expert_probs, dim=1)  # [num_experts, H, W]

        # Initialize output
        final_probs = torch.zeros((C, H, W), dtype=expert_probs.dtype, device=device)

        # Create masks for each expert's specialized class
        expert_masks = torch.zeros((num_experts, H, W), dtype=torch.bool, device=device)
        for expert_idx in range(num_experts):
            expert_masks[expert_idx] = predictions[expert_idx] == expert_idx

        # Find regions
        overlap_count = torch.sum(expert_masks, dim=0)  # [H, W]
        exact_match = overlap_count == 1
        no_expert_match = overlap_count == 0
        conflict = overlap_count > 1

        # Handle exact matches
        for expert_idx in range(num_experts):
            mask = expert_masks[expert_idx] & exact_match  # [H, W]
            if mask.any():
                mask_expanded = mask.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]
                final_probs[mask_expanded] = expert_probs[expert_idx][mask_expanded]

        # Handle no expert match
        if no_expert_match.any():
            mask_expanded = no_expert_match.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]
            final_probs[mask_expanded] = default_prob[mask_expanded]

        # Handle conflicts using highest confidence strategy
        if conflict.any():
            max_probs = torch.full(
                (H, W), -1.0, dtype=expert_probs.dtype, device=device
            )
            selected_experts = torch.full((H, W), -1, dtype=torch.long, device=device)

            for expert_idx in range(num_experts):
                current_mask = expert_masks[expert_idx] & conflict
                if current_mask.any():
                    confidence = expert_probs[expert_idx, expert_idx]  # [H, W]
                    # Only update where we have conflicts and higher confidence
                    update_mask = current_mask & (confidence > max_probs)
                    max_probs[update_mask] = confidence[update_mask]
                    selected_experts[update_mask] = expert_idx

            # Apply selected expert probabilities
            valid_selections = selected_experts >= 0
            if valid_selections.any():
                for expert_idx in range(num_experts):
                    expert_mask = (selected_experts == expert_idx) & valid_selections
                    if expert_mask.any():
                        mask_expanded = expert_mask.unsqueeze(0).expand(C, -1, -1)
                        final_probs[mask_expanded] = expert_probs[expert_idx][
                            mask_expanded
                        ]

        # Add batch dimension back
        return final_probs.unsqueeze(0)  # [1, C, H, W]

    def encode_decode(self, img, meta_data):
        """
        Basic encode_decode function that processes a single template
        Args:
            img: Input image
            meta_data: Image metadata
        """
        masks = self.model(
            img, self.template_dict
        )  # Pass the template dictionary directly
        if self.mode == "fusion":
            if hasattr(self, "dataset") and self.dataset in ["cocostuff", "ade20k"]:
                # if hasattr(self, "dataset") and self.dataset in ["ade20k"]:
                # Memory-optimized processing for cocostuff and ade20k datasets
                batch_size = (
                    16  # Can increase batch size since we're using half precision
                )
                num_masks = len(masks)
                num_batches = (num_masks + batch_size - 1) // batch_size

                # Initialize tensors for expert and default predictions
                expert_probs = []
                default_prob = None

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_masks)

                    batch_final_masks = []
                    for mask in masks[start_idx:end_idx]:
                        # Resize mask and convert to half precision
                        resized_mask = resize(
                            input=mask,
                            size=img.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).half()  # Convert to float16
                        batch_final_masks.append(resized_mask)

                    # Stack batch and split into experts/default
                    batch_stacked = torch.stack(batch_final_masks)

                    if batch_idx == num_batches - 1 and end_idx == num_masks:
                        # Last batch contains the default model prediction
                        default_prob = batch_stacked[-1]
                        if len(batch_stacked) > 1:
                            expert_probs.append(batch_stacked[:-1])
                    else:
                        expert_probs.append(batch_stacked)

                # Simple concatenation of all expert predictions at once
                expert_probs = torch.cat(expert_probs, dim=0)

                # Apply expert fusion (convert back to float32 if needed)
                fused_probs = self.expert_fusion(
                    expert_probs=expert_probs,
                    default_prob=default_prob,
                )

                return fused_probs
            else:
                # Simple direct processing for other datasets
                # Resize all masks at once
                resized_masks = [
                    resize(
                        input=mask,
                        size=img.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for mask in masks
                ]

                # Stack all masks and split into expert and default predictions
                all_probs = torch.stack(resized_masks)
                expert_probs = all_probs[:-1]  # All but last
                default_prob = all_probs[-1]  # Last one is default

            # Apply expert fusion
            fused_probs = self.expert_fusion(
                expert_probs=expert_probs,
                default_prob=default_prob,
            )

            return fused_probs
        elif self.mode == "compute_metric":
            # if hasattr(self, "dataset") and self.dataset in ["cocostuff", "ade20k"]:
            #     # Process masks in batches to avoid memory overflow
            #     batch_size = 8  # Can adjust this value based on available GPU memory
            #     num_masks = masks.shape[0]
            #     num_batches = (num_masks + batch_size - 1) // batch_size

            #     # Initialize output tensor with the correct shape including channel dimension
            #     final_masks = torch.zeros(
            #         (num_masks, masks.shape[2], *img.shape[-2:]),
            #         device=masks.device,
            #         dtype=torch.float16,  # Use half precision for memory efficiency
            #     )

            #     for batch_idx in range(num_batches):
            #         start_idx = batch_idx * batch_size
            #         end_idx = min((batch_idx + 1) * batch_size, num_masks)

            #         # Process current batch
            #         batch_masks = masks[start_idx:end_idx]
            #         batch_resized = resize(
            #             input=batch_masks.squeeze(1),
            #             size=img.shape[-2:],
            #             mode="bilinear",
            #             align_corners=False,
            #         ).half()  # Convert to half precision

            #         # Store directly in the pre-allocated tensor
            #         final_masks[start_idx:end_idx] = batch_resized

            #         # Clear intermediate tensors
            #         del batch_masks
            #         del batch_resized
            #         self._force_gpu_cleanup()

            #     return final_masks
            # else:
            masks = resize(
                input=masks.squeeze(1),
                size=img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return masks
        else:
            masks = resize(
                input=masks,
                size=img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return masks

    def inference(
        self,
        inputs: Tensor,
        batch_img_metas: List[dict],
    ) -> Tensor:
        """Inference with slide/whole style and template iteration."""

        assert self.test_cfg.get("mode", "whole") in ["slide", "whole"]

        if self.test_cfg.mode == "slide":
            if self.mode == "compute_metric":
                seg_logits = self.slide_inference(
                    inputs, batch_img_metas, num_templates=self.num_templates
                )
            else:
                seg_logits = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logits = self.whole_inference(inputs, batch_img_metas)

        return seg_logits

    def slide_inference(
        self, inputs: Tensor, batch_img_metas: List[dict], num_templates=None
    ) -> Tensor:

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        if num_templates is not None:
            batch_size = num_templates
        out_channels = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # Use half precision only for specific datasets
        use_half = hasattr(self, "dataset") and self.dataset in ["cocostuff", "ade20k"]
        dtype = (
            torch.float16
            if use_half and self.mode == "compute_metric"
            else torch.float32
        )  # Use half only for compute_metric on specific datasets

        # Initialize tensors with appropriate precision
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img), dtype=dtype)
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img), dtype=dtype)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]["img_shape"] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                # Convert to half precision if needed
                if use_half:
                    crop_seg_logit = crop_seg_logit.half()
                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1

                # Force GPU cleanup after each patch
                # self._force_gpu_cleanup()

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`] or Tensor: Segmentation results.
            For compute_metric mode, returns the raw logits tensor.
            For other modes, returns SegDataSample containing predictions.
        """
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        if self.mode == "compute_metric":
            return seg_logits
        else:
            return self.postprocess_result(seg_logits, data_samples)

    def resize_predictions(
        self, seg_logits: Tensor, data_samples: OptSampleList = None
    ) -> Tensor:
        """Resize the predictions to original image size.

        Args:
            seg_logits (Tensor): The segmentation logits with shape (num_templates+1, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data samples
                containing metadata about original image sizes.

        Returns:
            Tensor: Resized logits with shape (num_templates+1, C, original_H, original_W).
        """
        num_templates_plus_one, C, H, W = seg_logits.shape

        if data_samples is None:
            return seg_logits

        resized_logits = []
        for template_idx in range(num_templates_plus_one):
            template_logits = seg_logits[
                template_idx : template_idx + 1
            ]  # Keep batch dim
            img_meta = data_samples[
                0
            ].metainfo  # Use first sample's metadata as they're all same

            # Resize to original shape
            template_logits = resize(
                template_logits,
                size=img_meta["ori_shape"],
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

            resized_logits.append(template_logits.squeeze(0))  # Remove batch dim

        # Stack all resized predictions
        return torch.stack(
            resized_logits
        )  # Shape: (num_templates+1, C, original_H, original_W)
