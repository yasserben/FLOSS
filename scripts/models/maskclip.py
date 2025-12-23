"""
MaskCLIP Feature Extractor - Wrapper around existing implementation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import mmseg.models to register all components (including SegDataPreProcessor)
import mmseg.models  # This registers SegDataPreProcessor in mmengine registry

# Import the existing MaskCLIP implementation
from mmseg.models.segmentors.maskclip import MaskClip, MaskClipHead
from mmseg.models.segmentors.maskclip.utils.prompt_templates import imagenet_templates


class MaskCLIPExtractor:
    """
    Feature extractor wrapper for MaskCLIP.
    Uses the existing MaskClip and MaskClipHead classes directly.
    """

    def __init__(
        self,
        class_names: List[str],
        device: str = "cuda",
        clip_model: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
    ):
        self.class_names = class_names
        self.device = device
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.templates = imagenet_templates
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MaskCLIP model using the existing implementation."""
        print(f"📥 Loading MaskCLIP ({self.clip_model})...")

        # Build MaskClip model with the same config as in configs/_base_/models/maskclip.py
        self.model = MaskClip(
            backbone=dict(img_size=224, patch_size=16),
            decode_head=dict(
                type="MaskClipHead",
                in_channels=768,
                text_channels=512,
                pretrained=self.pretrained,
                mode="compute_metric",  # This enables per-template computation
                id_start=0,
                id_end=79,
            ),
            clip_model=self.clip_model,
            class_names=self.class_names,
        )
        self.model.eval().to(self.device)

        print(f"✅ MaskCLIP loaded with {len(self.class_names)} classes!")

    @torch.no_grad()
    def compute_text_features(self) -> Dict[str, torch.Tensor]:
        """
        Compute text features using MaskClipHead's _embed_templates method.

        Returns:
            Dictionary with per_template and averaged features
        """
        print(f"🔤 Computing text features for {len(self.class_names)} classes...")

        decode_head = self.model.decode_head

        per_template_features = []
        for template_id in tqdm(range(80), desc="Computing text features"):
            # Use the existing _embed_templates method
            template_features = decode_head._embed_templates(
                decode_head.model,
                self.class_names,
                template_id,
                self.device,
            )
            per_template_features.append(template_features.cpu())

        per_template_features = torch.stack(per_template_features)

        # Compute averaged features
        averaged_features = per_template_features.mean(dim=0)
        averaged_features = averaged_features / averaged_features.norm(
            dim=-1, keepdim=True
        )

        return {
            "per_template": per_template_features,
            "averaged": averaged_features,
            "class_names": self.class_names,
            "templates": self.templates,
        }

    @torch.no_grad()
    def compute_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute dense vision features using MaskClip's extract_feat.

        Args:
            image: Image tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Dense features [B, text_dim, h, w]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Use the existing extract_feat method
        # First apply CLIP normalization
        image = self.model.clip_T(image)
        feat = self.model.extract_feat(image)

        # Project to text space
        feat = self.model.decode_head.proj(feat)
        feat = feat / feat.norm(dim=1, keepdim=True)

        return feat.cpu()

    @property
    def model_name(self) -> str:
        return "maskclip"
