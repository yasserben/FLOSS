import logging
import sys

import torch
import torch.nn as nn
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from enum import Enum
from .clip import load, tokenize
from .model import *
from .pamr import PAMR
from .prompts.imagenet_template import openai_imagenet_template
import json

sys.path.append("..")


@MODELS.register_module()
class NACLIP(BaseSegmentor):
    def __init__(
        self,
        clip_path,
        name_path,
        device=torch.device("cuda"),
        arch="reduced",
        attn_strategy="naclip",
        gaussian_std=5.0,
        pamr_steps=0,
        pamr_stride=(8, 16),
        prob_thd=0.0,
        logit_scale=100,
        slide_stride=112,
        slide_crop=224,
        template_id=None,
        top_n=4,
        mode="default",
        dataset="cityscapes",
        id_start=0,
        id_end=79,
    ):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True,
        )
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _ = load(clip_path, device=device, jit=False)

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.mode = mode
        self.dataset = dataset
        self.id_start = id_start
        self.id_end = id_end

        if self.mode == "fusion":
            self.template_dict = self._load_template_rankings(
                top_n=top_n,
                dataset=dataset,
            )
        else:
            self.template_dict = None

        if self.mode == "compute_metric":
            self.num_templates = self.id_end - self.id_start + 1

        self.query_features = self.encode_text_features(
            query_words, device, template_id, self.template_dict
        )

        self.dtype = self.query_features.dtype
        self.net.visual.set_params(arch, attn_strategy, gaussian_std)
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False
        self.pamr = (
            PAMR(pamr_steps, dilations=pamr_stride).to(device)
            if pamr_steps > 0
            else None
        )

        logging.info(
            f"attn_strategy is {attn_strategy}, arch is {arch} & Gaussian std is {gaussian_std}"
        )

    def _load_template_rankings(
        self,
        top_n=1,
        dataset="cityscapes",
    ):
        """Load and process template rankings based on specified metric and threshold.

        Args:
            selection_mode (str): Metric to rank templates by ('iou' or 'entropy')

        Returns:
            dict: Dictionary mapping class names to their best template IDs
        """
        # Load the json file
        with open(
            f"rankings/naclip/template_rankings_naclip_{dataset}.json",
            "r",
        ) as f:
            rankings_data = json.load(f)

        # Validate selection mode and construct ranking field
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
                    raise ValueError(
                        f"No valid templates found with cap_pp={cap_pp} for class {class_name}"
                    )

        # Apply dataset-specific fixes
        if self.dataset == "pascalvoc20":
            template_dict = self._apply_quick_and_dirty_fix(template_dict)

        return template_dict

    def _apply_quick_and_dirty_fix(self, template_dict):
        """Apply quick and dirty fixes for pascalvoc20 dataset template mappings.

        Args:
            template_dict (dict): Template dictionary to modify

        Returns:
            dict: Modified template dictionary
        """
        # replace the key 'boat' by 'ship'
        template_dict["ship"] = template_dict["boat"]
        del template_dict["boat"]

        # replace the key "diningtable" by "table"
        template_dict["table"] = template_dict["diningtable"]
        del template_dict["diningtable"]

        # add the keys "person in shirt", person in jeans, person in dress, person in sweater, person in skirt, person in jacket" "
        template_dict["person in shirt"] = template_dict["person"]
        template_dict["person in jeans"] = template_dict["person"]
        template_dict["person in dress"] = template_dict["person"]
        template_dict["person in sweater"] = template_dict["person"]
        template_dict["person in skirt"] = template_dict["person"]
        template_dict["person in jacket"] = template_dict["person"]

        # add the keys 'television monitor", 'tv monitor" and "monitor"
        template_dict["television monitor"] = template_dict["tvmonitor"]
        template_dict["tv monitor"] = template_dict["tvmonitor"]
        template_dict["monitor"] = template_dict["tvmonitor"]
        template_dict["television"] = template_dict["tvmonitor"]
        template_dict["screen"] = template_dict["tvmonitor"]

        return template_dict

    def expert_fusion(
        self,
        expert_probs: torch.Tensor,  # [num_experts, 1, C, H, W]
        default_prob: torch.Tensor,  # [1, C, H, W]
    ) -> torch.Tensor:
        """
        Merge predictions using expert fusion with different strategies, keeping probability vectors.
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
        dtype = expert_probs.dtype

        # Get predictions from probabilities
        predictions = torch.argmax(expert_probs, dim=1)  # [num_experts, H, W]

        # Initialize output with same dtype as input
        final_probs = torch.zeros((C, H, W), device=device, dtype=dtype)

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
                final_probs[mask_expanded] = expert_probs[expert_idx][mask_expanded].to(
                    dtype
                )

        # Handle no expert match
        if no_expert_match.any():
            mask_expanded = no_expert_match.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]
            final_probs[mask_expanded] = default_prob[mask_expanded].to(dtype)

        # Handle conflicts based on strategy
        if conflict.any():
            max_probs = torch.full((H, W), -1.0, device=device, dtype=dtype)
            selected_experts = torch.full((H, W), -1, dtype=torch.long, device=device)

            for expert_idx in range(num_experts):
                current_mask = expert_masks[expert_idx] & conflict
                if current_mask.any():
                    confidence = expert_probs[expert_idx, expert_idx].to(
                        dtype
                    )  # [H, W]
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
                        ].to(dtype)

        # Add batch dimension back
        return final_probs.unsqueeze(0)  # [1, C, H, W]

    def encode_text_features(
        self, query_words, device, template_id=None, template_dict=None
    ):
        if self.mode == "fusion":
            # Create list to store expert embeddings
            list_of_experts = []

            # Create expert embeddings for each class
            for name in query_words:
                expert_embeddings = []
                # Get templates for current class
                if isinstance(template_dict[name], list):
                    templates = [
                        openai_imagenet_template[tid] for tid in template_dict[name]
                    ]
                else:
                    templates = [openai_imagenet_template[template_dict[name]]]

                # Get embeddings for all classes using current class's templates
                for element in query_words:
                    all_prompts = [
                        tokenize([template(element)]).to(device)
                        for template in templates
                    ]
                    # Stack all prompts
                    all_prompts = torch.cat(all_prompts)

                    with torch.no_grad():
                        feature = self.net.encode_text(all_prompts)
                        feature /= feature.norm(dim=-1, keepdim=True)
                        feature = feature.mean(dim=0)
                        feature /= feature.norm()
                    expert_embeddings.append(feature)

                # Stack embeddings for current expert
                expert_features = torch.stack(expert_embeddings)
                list_of_experts.append(expert_features)

            # Stack default embeddings and add to experts list
            default_embeddings = []
            for qw in query_words:
                if template_id is not None:
                    query = tokenize(
                        [openai_imagenet_template[int(template_id)](qw)]
                    ).to(device)
                else:
                    query = tokenize(
                        [temp(qw) for temp in openai_imagenet_template]
                    ).to(device)
                with torch.no_grad():
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    default_embeddings.append(feature)

            # Stack default embeddings and add to experts list
            default_features = torch.stack(default_embeddings)
            list_of_experts.append(default_features)

            # Stack all experts into final tensor
            return torch.stack(
                list_of_experts
            )  # [num_experts+1, num_classes, text_channels]

        elif self.mode == "compute_metric":
            # Create list to store template embeddings for specified range
            list_of_templates = []

            # Create embeddings only for templates in the specified range
            for template_idx in range(self.id_start, self.id_end + 1):
                template_embeddings = []
                for qw in query_words:
                    query = tokenize([openai_imagenet_template[template_idx](qw)]).to(
                        device
                    )
                    with torch.no_grad():
                        feature = self.net.encode_text(query)
                        feature /= feature.norm(dim=-1, keepdim=True)
                        feature = feature.mean(dim=0)
                        feature /= feature.norm()
                        template_embeddings.append(feature)
                template_features = torch.stack(template_embeddings)
                list_of_templates.append(template_features)

            # Stack all templates into final tensor
            return torch.stack(
                list_of_templates
            )  # [num_templates_in_range, num_classes, text_channels]

        else:
            query_features = list()
            with torch.no_grad():
                for qw in query_words:
                    if template_id is not None:
                        # Use only the specified template
                        query = tokenize(
                            [openai_imagenet_template[int(template_id)](qw)]
                        ).to(device)
                    else:
                        # Use all templates
                        query = tokenize(
                            [temp(qw) for temp in openai_imagenet_template]
                        ).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            return torch.cat(query_features, dim=0)

    def forward_feature(self, img):
        if type(img) == list:
            img = img[0]

        image_features = self.net.encode_image(img, return_all=True)
        image_features = image_features[:, 1:]
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if self.mode == "fusion":
            num_experts, num_classes, text_channels = self.query_features.shape
            reshaped_query_features = self.query_features.reshape(-1, text_channels)

            patch_size = self.net.visual.patch_size
            w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
            out_dim = image_features.shape[-1]
            image_features = image_features.permute(0, 2, 1).reshape(-1, out_dim, w, h)

            outputs = F.conv2d(
                image_features, reshaped_query_features[:, :, None, None]
            )
            outputs = outputs.reshape(
                image_features.size(0),
                num_experts,
                num_classes,
                outputs.size(2),
                outputs.size(3),
            )
            outputs = F.softmax(outputs * self.logit_scale, dim=2)
            outputs = outputs.permute(1, 0, 2, 3, 4)

            final_outputs = []
            for output in outputs:
                output = nn.functional.interpolate(
                    output,
                    size=img.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                final_outputs.append(output)
            final_outputs = torch.stack(final_outputs)

            # Get the default model's prediction (last expert)
            default_prob = final_outputs[-1]
            # Remove default model from experts tensor
            expert_probs = final_outputs[:-1]

            # Apply expert fusion
            fused_probs = self.expert_fusion(
                expert_probs=expert_probs,
                default_prob=default_prob,
            )
            return fused_probs

        elif self.mode == "compute_metric":
            # Adapted logic from MaskClip cls_seg for compute_metric
            num_templates, num_classes, text_channels = self.query_features.shape
            # query_features are already [num_templates+1, num_classes, text_channels]
            reshaped_templates = self.query_features.reshape(-1, text_channels)

            # image_features are [B, N, C_img], need [B, C_img, w, h]
            patch_size = self.net.visual.patch_size
            w, h = img.shape[-2] // patch_size, img.shape[-1] // patch_size
            B, N, C_img = image_features.shape  # N = w * h
            image_features_reshaped = image_features.permute(0, 2, 1).reshape(
                B, C_img, w, h
            )

            # Image features are already normalized in NACLIP's encode_image
            image_features_norm = image_features_reshaped

            # Compute scores using conv2d
            output = F.conv2d(image_features_norm, reshaped_templates[:, :, None, None])

            # Reshape output to [B, num_templates, num_classes, H_feat, W_feat]
            output = output.reshape(
                B, num_templates, num_classes, output.size(2), output.size(3)
            )

            # NACLIP applies logit_scale later, MaskClip applies tau & softmax here.
            # For consistency with NACLIP's structure, return raw logits here.
            # Interpolation is also handled later in predict().

            # output = F.softmax(output * self.logit_scale, dim=2)
            # Permute to put templates first: [num_templates, B, num_classes, H_feat, W_feat]
            output = output.permute(1, 0, 2, 3, 4)

            # Interpolate output
            output = nn.functional.interpolate(
                output.squeeze(1),
                size=img.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            return output

        else:  # Default mode (original logic, but without interpolation)
            logits = image_features @ self.query_features.T

            patch_size = self.net.visual.patch_size
            w, h = (
                img.shape[-2] // patch_size,
                img.shape[-1] // patch_size,
            )  # Use img shape directly
            B, N, C_img = image_features.shape
            out_dim = logits.shape[-1]  # Should be num_classes
            # Reshape from [B, N, C_cls] to [B, C_cls, w, h]
            logits = logits.permute(0, 2, 1).reshape(B, out_dim, w, h)

            logits = nn.functional.interpolate(
                logits,
                size=img.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            return logits

    def forward_slide(self, img, stride=112, crop_size=224, num_templates=None):
        """
        Inference by sliding-window with overlap. If h_crop > h_img or w_crop > w_img,
        the small patch will be used to decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        if num_templates is not None:
            batch_size = num_templates
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # Determine dtype based on dataset for memory optimization
        use_half = hasattr(self, "dataset") and self.dataset in ["cocostuff", "ade20k"]
        dtype = (
            torch.float16
            if use_half and self.mode == "compute_metric"
            else torch.float32
        )  # Use half only for compute_metric on specific datasets

        preds = img.new_zeros((batch_size, out_channels, h_img, w_img), dtype=dtype)
        count_mat = img.new_zeros(
            (batch_size, 1, h_img, w_img), dtype=dtype
        )  # Match preds dtype

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                # Convert to half precision if needed
                if use_half:
                    crop_seg_logit = crop_seg_logit.half()
                preds += nn.functional.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        logits = preds / count_mat
        return logits

    def predict(self, inputs, data_samples):
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

        if self.slide_crop > 0:
            if self.mode == "compute_metric":
                seg_logits = self.forward_slide(
                    inputs,
                    self.slide_stride,
                    self.slide_crop,
                    num_templates=self.num_templates,
                )
            else:
                seg_logits = self.forward_slide(
                    inputs, self.slide_stride, self.slide_crop
                )
        else:
            seg_logits = self.forward_feature(inputs)

        img_size = batch_img_metas[0]["ori_shape"]

        # # Handle large datasets with batch processing for interpolation
        # if hasattr(self, "dataset") and self.dataset in ["cocostuff", "ade20k"]:
        #     # Process interpolation in batches to avoid memory overflow
        #     batch_size = 2  # Can adjust based on available GPU memory
        #     num_masks = seg_logits.shape[0]
        #     num_batches = (num_masks + batch_size - 1) // batch_size

        #     # Initialize output tensor with correct shape
        #     final_logits = torch.zeros(
        #         (num_masks, seg_logits.shape[1], *img_size),
        #         device=seg_logits.device,
        #         dtype=torch.float16,  # Use half precision for memory efficiency
        #     )

        #     for batch_idx in range(num_batches):
        #         start_idx = batch_idx * batch_size
        #         end_idx = min((batch_idx + 1) * batch_size, num_masks)

        #         # Process current batch
        #         batch_logits = seg_logits[start_idx:end_idx]
        #         batch_resized = nn.functional.interpolate(
        #             batch_logits,
        #             size=img_size,
        #             mode="bilinear",
        #             align_corners=self.align_corners,
        #         ).half()  # Convert to half precision

        #         # Store in pre-allocated tensor
        #         final_logits[start_idx:end_idx] = batch_resized

        #         # Clear intermediate tensors
        #         del batch_logits
        #         del batch_resized
        #         torch.cuda.empty_cache()

        #     seg_logits = final_logits
        # else:
        # Original interpolation for other datasets
        seg_logits = nn.functional.interpolate(
            seg_logits,
            size=img_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if self.pamr:
            img = nn.functional.interpolate(
                inputs, size=img_size, mode="bilinear", align_corners=self.align_corners
            )
            try:
                seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)
            except RuntimeError as e:
                logging.warning(
                    f"Couldn't apply PAMR for image {batch_img_metas[0]['img_path'].split('/')[-1]} "
                    f'of size {img_size}, probably due to low memory. Error message: "{str(e)}"'
                )

        if self.mode == "compute_metric":
            # Unlike CLIPDinoiser and MaskClip, NACLIP applies logit_scale + softmax here => after interpolation to original size
            seg_logits = F.softmax(seg_logits * self.logit_scale, dim=1)
            # Handle case where num_cls != num_queries without iteration
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                cls_index = nn.functional.one_hot(
                    self.query_idx
                )  # [num_queries, num_cls]
                cls_index = cls_index.T  # [num_cls, num_queries]
                cls_index = cls_index.view(
                    1, num_cls, num_queries, 1, 1
                )  # Add dims for broadcasting
                seg_logits = seg_logits.unsqueeze(1)  # [B, 1, num_queries, H, W]
                seg_logits = (seg_logits * cls_index).max(2)[0]  # [B, num_cls, H, W]

            return seg_logits
        else:
            return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_probs = torch.softmax(
                seg_logits[i] * self.logit_scale, dim=0
            )  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_probs = seg_probs.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_probs = (seg_probs * cls_index).max(1)[0]

            seg_pred = seg_probs.argmax(0, keepdim=True)
            # seg_pred[seg_probs.max(0, keepdim=True)[0] < self.prob_thd] = 0
            # seg_probs /= seg_probs.sum(0, keepdim=True)

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_probs}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred}),
                }
            )

        return data_samples

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(", ")
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "") for item in class_names]
    return class_names, class_indices
