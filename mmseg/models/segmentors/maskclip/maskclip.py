import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

# from mmseg.ops import resize
# Yasser
from mmseg.models.utils import resize

from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer, create_model_from_pretrained
from mmseg.registry import MODELS
import torchvision.transforms as T
from .utils.prompt_templates import imagenet_templates


OPENAI_NORMALIZE = T.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


@MODELS.register_module()
class MaskClip(nn.Module):
    def __init__(self, backbone, decode_head, clip_model, class_names):
        super(MaskClip, self).__init__()

        self.decode_head = eval(decode_head.get("type"))(
            clip_model, class_names, **decode_head
        )
        self.patch_size = backbone.get("patch_size")
        self.img_size = tuple([backbone.get("img_size", 224)] * 2)
        pretrained = decode_head.get("pretrained")
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model

        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output

        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(
            hook_fn_forward
        )
        self._positional_embd = nn.Parameter(
            self.backbone.visual.positional_embedding.data.clone()
        )

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0] * hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if (
                pos_len
                == (self.img_size[0] // self.patch_size)
                * (self.img_size[1] // self.patch_size)
                + 1
            ):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError("{}, {}".format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape, (pos_h, pos_w), "bicubic"
            )[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(
            1, 0, 2
        )
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = torch.nn.functional.linear(
            y, block.attn.in_proj_weight, block.attn.in_proj_bias
        )
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)

        pos_embed_weight = resize(
            input=pos_embed_weight, size=input_shpae, align_corners=False, mode=mode
        )

        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: Tensor, template_dict: dict, return_feat=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        with torch.no_grad():
            inputs = self.clip_T(inputs)
            x = self.extract_feat(inputs)
        if return_feat:
            feats = self.decode_head(
                inputs=x, template_dict=template_dict, return_feat=return_feat
            )
            return feats
        else:
            output = self.decode_head(x, template_dict)
            return output
        # if return_feat:
        #     seg_logits, feats = self.decode_head(x, template_id, return_feat)
        #     return seg_logits, feats
        # else:
        #     seg_logits = self.decode_head(x, template_id)
        # return seg_logits


class MaskClipHead(nn.Module):
    """
    This class is a modified version of the MaskClipHead class.
    It initializes the class embeddings using a specified template ID during each forward pass.
    """

    def __init__(
        self,
        clip_model,
        class_names,
        in_channels=3,
        text_channels=512,
        use_templates=False,
        id_template=None,
        pretrained=None,
        tau: float = 0.01,
        mode="default",
        id_start=0,
        id_end=79,
        **kwargs,
    ):
        super(MaskClipHead, self).__init__()

        self.text_channels = text_channels
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.class_names = class_names
        self.in_channels = in_channels
        self.id_template = id_template
        self.tokenizer = get_tokenizer(clip_model)
        self.model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        self.model.eval()
        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(self.model.visual.proj.t()[:, :, None, None])
        self.class_embeddings = None  # Initialize as None
        self.first_time = True
        self.tau = tau
        self.mode = mode
        self.id_start = id_start
        self.id_end = id_end

    @torch.no_grad()
    def _embed_label(
        self,
        text_model: torch.nn.Module,
        label: str,
        template_dict: dict,
        device: torch.device,
    ) -> torch.Tensor:

        all_templates = imagenet_templates

        if template_dict is not None:
            if isinstance(template_dict[label], list):
                # Handle list of template IDs
                templates = [all_templates[tid] for tid in template_dict[label]]
            else:
                # Handle single template ID
                templates = [all_templates[template_dict[label]]]
        else:
            templates = all_templates

        if self.id_template is not None:
            if isinstance(self.id_template, (list, tuple)):
                templates = [all_templates[tid] for tid in self.id_template]
            else:
                templates = [all_templates[self.id_template]]

        all_prompts = [self.tokenizer(template.format(label)) for template in templates]
        # put all prompts into a tensor
        all_prompts = torch.cat(all_prompts)
        # put to GPU
        all_prompts = all_prompts.to(device)
        out = text_model.encode_text(all_prompts)
        out /= out.norm(dim=-1, keepdim=True)
        out = out.mean(dim=0)
        return out

    @torch.no_grad()
    def _embed_experts(
        self,
        text_model: torch.nn.Module,
        class_name: str,
        list_of_classes: List[str],
        template_dict: dict,
        device: torch.device,
    ) -> torch.Tensor:

        all_templates = imagenet_templates

        if template_dict is not None:
            if isinstance(template_dict[class_name], list):
                # Handle list of template IDs
                templates = [
                    imagenet_templates[tid] for tid in template_dict[class_name]
                ]
            elif isinstance(template_dict[class_name], int):
                # Handle single template ID
                templates = [imagenet_templates[template_dict[class_name]]]
            elif template_dict[class_name] is None:
                templates = imagenet_templates
        else:
            templates = imagenet_templates

        all_embeddings = []
        for element in list_of_classes:
            all_prompts = [
                self.tokenizer(template.format(element)) for template in templates
            ]
            # put all prompts into a tensor
            all_prompts = torch.cat(all_prompts)
            # put to GPU
            all_prompts = all_prompts.to(device)
            out = text_model.encode_text(all_prompts)
            out /= out.norm(dim=-1, keepdim=True)
            out = out.mean(dim=0)
            all_embeddings.append(out)
        return torch.stack(all_embeddings)

    @torch.no_grad()
    def _embed_templates(
        self,
        text_model: torch.nn.Module,
        list_of_classes: List[str],
        template_id: int,
        device: torch.device,
    ) -> torch.Tensor:

        if template_id is not None:
            template = imagenet_templates[template_id]
            all_prompts = [
                self.tokenizer(template.format(element)) for element in list_of_classes
            ]
        else:
            template = imagenet_templates
            all_prompts = [
                self.tokenizer(t.format(element))
                for t in template
                for element in list_of_classes
            ]
        # put all prompts into a tensor
        all_prompts = torch.cat(all_prompts)
        # put to GPU
        all_prompts = all_prompts.to(device)
        out = text_model.encode_text(all_prompts)
        out /= out.norm(dim=-1, keepdim=True)
        return out

    def _get_class_embeddings(
        self,
        text_model: torch.nn.Module,
        class_names: List[str],
        template_dict: dict,
        device: torch.device,
    ):
        if self.mode == "fusion":
            list_of_experts = []
            for name in class_names:
                list_of_experts.append(
                    self._embed_experts(
                        text_model, name, class_names, template_dict, device
                    )
                )
            # Add the default model embedding
            list_of_experts.append(
                self._embed_experts(text_model, "default", class_names, None, device)
            )
            return torch.stack(list_of_experts)
        elif self.mode == "compute_metric":
            list_of_templates = []
            for template_id in range(self.id_start, self.id_end + 1):
                list_of_templates.append(
                    self._embed_templates(text_model, class_names, template_id, device)
                )
            return torch.stack(list_of_templates)
        else:
            aug_embeddings = torch.stack(
                [
                    self._embed_label(text_model, label, template_dict, device)
                    for label in class_names
                ]
            )
            aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
            return aug_embeddings.squeeze(1)

    @torch.no_grad()
    def forward(self, inputs, template_dict, return_feat=False):
        if self.first_time:
            self.class_embeddings = self._get_class_embeddings(
                self.model, self.class_names, template_dict, device=inputs.device
            )
            self.first_time = False

        self.class_embeddings_after = self.class_embeddings

        v = inputs
        feat = self.proj(v)
        if return_feat:
            return feat
        else:
            output = self.cls_seg(feat)
            return output

    @torch.no_grad()
    def cls_seg(self, feat):
        if self.mode == "fusion":
            feat = feat / feat.norm(dim=1, keepdim=True)
            # Compute all expert predictions in a single conv2d operation
            num_experts, num_classes, text_channels = self.class_embeddings_after.shape
            reshaped_experts = self.class_embeddings_after.reshape(-1, text_channels)
            output = F.conv2d(feat, reshaped_experts[:, :, None, None])
            output = output.reshape(
                feat.size(0), num_experts, num_classes, output.size(2), output.size(3)
            )
            output = F.softmax(output / self.tau, dim=2)
            return output.permute(
                1, 0, 2, 3, 4
            )  # (num_experts, batch_size, num_classes, H, W)

        elif self.mode == "compute_metric":
            feat = feat / feat.norm(dim=1, keepdim=True)
            # Compute all template predictions in a single conv2d operation
            num_templates, num_classes, text_channels = (
                self.class_embeddings_after.shape
            )
            reshaped_templates = self.class_embeddings_after.reshape(-1, text_channels)
            output = F.conv2d(feat, reshaped_templates[:, :, None, None])
            output = output.reshape(
                feat.size(0), num_templates, num_classes, output.size(2), output.size(3)
            )
            output = F.softmax(output / self.tau, dim=2)
            return output.permute(
                1, 0, 2, 3, 4
            )  # (num_templates, batch_size, num_classes, H, W)
        else:
            feat = feat / feat.norm(dim=1, keepdim=True)
            output = F.conv2d(feat, self.class_embeddings_after[:, :, None, None])
            output = F.softmax(output / self.tau, dim=1)
            return output
