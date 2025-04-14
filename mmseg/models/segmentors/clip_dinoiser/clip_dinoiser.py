import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from typing import List
from mmengine.model import BaseModel
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.segmentors.base import BaseSegmentor

from omegaconf import OmegaConf
from mmseg.registry import MODELS
from mmseg.models import build_model, build_model_custom

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@MODELS.register_module()
class DinoCLIP(BaseModel):
    """
    Base model for all the backbones. Implements CLIP features refinement based on DINO dense features and background
    refinement using FOUND model.
    """

    def __init__(
        self,
        clip_backbone,
        class_names,
        data_preprocessor,
        init_cfg,
        vit_arch="vit_base",
        vit_patch_size=16,
        enc_type_feats="k",
        gamma=0.2,
        delta=0.99,
        apply_found=False,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.vit_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.gamma = gamma
        self.vit_patch_size = vit_patch_size
        self.apply_found = apply_found
        self.delta = delta

        # ==== build MaskCLIP backbone directly from config =====
        self.clip_backbone = build_model_custom(clip_backbone, class_names=class_names)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def get_clip_features(self, x: torch.Tensor, template_dict: dict):
        """
        Gets MaskCLIP features
        :param x: (torch.Tensor) - batch of input images
        :param template_id: (int) - ID of the template to use
        :return: (torch.Tensor) - clip dense features, (torch.Tensor) - output probabilities
        """
        x = self.make_input_divisible(x)
        # maskclip_map, feat = self.clip_backbone(x, template_id, return_feat=True)
        feat = self.clip_backbone(x, template_dict, return_feat=True)
        return feat

    def get_found_preds(self, x: torch.Tensor, resize=None):
        """
        Gets FOUND predictions.
        :param x: (torch.Tensor) - batch of input images
        :param resize: optional (tuple(int)) - size to resize the output prediction map to
        :return: (torch.Tensor) - saliency predictions
        """
        x = self.make_input_divisible(x)
        x = self.dino_T(x)
        found_preds, _, shape_f, att = self.found_model.forward_step(x, for_eval=True)
        preds = torch.sigmoid(found_preds.detach()).float()
        if resize is not None:
            preds = T.functional.resize(preds, resize)
        return preds

    @staticmethod
    def compute_weighted_pool(maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        """
        Weighted pooling method.
        :param maskclip_feats: torch.tensor - raw clip features
        :param corrs: torch.tensor - correlations as weights for pooling mechanism
        :return: torch.tensor - refined clip features
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = resize(
                input=maskclip_feats,
                size=(h_w, w_w),
                mode="bilinear",
                align_corners=False,
            )
            h_m, w_m = h_w, w_w

        maskclip_feats_ref = torch.einsum(
            "bnij, bcij -> bcn", corrs, maskclip_feats
        )  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

        # RESHAPE back to 2d
        maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
        return maskclip_feats_ref


@MODELS.register_module()
class CLIP_DINOiser(DinoCLIP):
    """
    CLIP-DINOiser - torch.nn.Module with two single conv layers for object correlations (obj_proj) and background
    filtering (bkg_decoder).
    """

    def __init__(
        self,
        clip_backbone,
        class_names,
        data_preprocessor,
        init_cfg,
        vit_arch="vit_base",
        vit_patch_size=16,
        enc_type_feats="v",
        feats_idx=-3,
        gamma=0.2,
        delta=0.99,
        in_dim=256,
        conv_kernel=3,
    ):
        # Initialize DinoCLIP
        DinoCLIP.__init__(
            self,
            clip_backbone,
            class_names,
            data_preprocessor,
            init_cfg,
            vit_arch,
            vit_patch_size,
            enc_type_feats,
            gamma,
        )

        in_size = 768 if feats_idx != "final" else 512
        self.gamma = gamma
        self.feats_idx = feats_idx
        self.delta = delta
        self.in_dim = in_dim
        self.clip_backbone_cfg = clip_backbone
        self.bkg_decoder = nn.Conv2d(in_size, 1, (1, 1))
        self.obj_proj = nn.Conv2d(
            in_size,
            in_dim,
            (conv_kernel, conv_kernel),
            padding=conv_kernel // 2,
            padding_mode="replicate",
        )

        # setup clip features for training
        if feats_idx != "final":
            train_feats = {}

            def get_activation(name):
                def hook(model, input, output):
                    train_feats[name] = output.detach().permute(
                        1, 0, 2
                    )  # change to batch first

                return hook

            self.clip_backbone.backbone.visual.transformer.resblocks[
                feats_idx
            ].ln_2.register_forward_hook(get_activation("clip_inter"))
            self.train_feats = train_feats

    def forward_pass(self, x: torch.Tensor, template_dict: dict):
        """Process a single template"""
        x = self.make_input_divisible(x)
        # Normally self.get_clip_features() gives two outputs, which are the features of CLIP
        # and the segmentation map given by MaskCLIP but we only need the first one
        # # clip_proj_feats = self.get_clip_features(x, template_id)[0]
        clip_proj_feats = self.get_clip_features(x, template_dict)
        B, c_dim, h, w = clip_proj_feats.shape
        if self.feats_idx != "final":
            clip_feats = self.train_feats["clip_inter"]
            B, N, c_dim = clip_feats.shape
            clip_feats = (
                clip_feats[
                    :,
                    1:,
                ]
                .permute(0, 2, 1)
                .reshape(B, c_dim, h, w)
            )
        else:
            clip_feats = clip_proj_feats
        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim, -1)
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)
        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(
            B, h * w, h, w
        )
        output = clip_feats / clip_feats.norm(dim=1, keepdim=True)
        bkg_out = self.bkg_decoder(output)

        return bkg_out, corrs, clip_proj_feats

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, template_dict: dict):
        """
        Forward pass with template dictionary mapping class names to their template IDs
        Args:
            inputs: Input tensor
            template_dict: Dictionary mapping class names to template IDs
        """
        preds, corrs, output = self.forward_pass(inputs, template_dict)
        B, C, hf, wf = output.shape
        preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False)

        # Compute weighted pooling --------------------------------------------------
        if self.gamma:
            corrs[corrs < self.gamma] = 0.0
        out_feats = self.compute_weighted_pool(output, corrs)

        # Get the predictions --------------------------------------------------

        output = self.clip_backbone.decode_head.cls_seg(out_feats)

        if self.apply_found:
            # Compute FOUND --------------------------------------------------
            soft_found = torch.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[
                :, 0, uncertain & (~r_hard_found.bool())
            ] = 1.0  # background class

        return output

    def extract_feat(self, inputs: torch.Tensor, template_dict: dict):
        """
        Forward pass with template dictionary mapping class names to their template IDs
        Args:
            inputs: Input tensor
            template_dict: Dictionary mapping class names to template IDs
        """
        preds, corrs, output = self.forward_pass(inputs, template_dict)
        B, C, hf, wf = output.shape
        preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False)

        # Compute weighted pooling --------------------------------------------------
        if self.gamma:
            corrs[corrs < self.gamma] = 0.0
        out_feats = self.compute_weighted_pool(output, corrs)

        return out_feats
