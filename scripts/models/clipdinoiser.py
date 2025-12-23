"""
CLIP-DINOiser Feature Extractor - Wrapper around existing implementation.
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
from mmseg.models.data_preprocessor import SegDataPreProcessor

# Import the existing CLIP-DINOiser implementation
from mmseg.models.segmentors.clip_dinoiser import CLIP_DINOiser
from mmseg.models.segmentors.maskclip.utils.prompt_templates import imagenet_templates


class CLIPDINOiserExtractor:
    """
    Feature extractor wrapper for CLIP-DINOiser.
    Uses the existing CLIP_DINOiser class directly with pretrained weights.
    """

    def __init__(
        self,
        class_names: List[str],
        device: str = "cuda",
        checkpoint_path: str = "checkpoints/clip_dinoiser/model_weights.pth",
    ):
        self.class_names = class_names
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.templates = imagenet_templates
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load CLIP-DINOiser model with pretrained weights."""
        print(f"📥 Loading CLIP-DINOiser...")

        # Build CLIP-DINOiser model with the same config as in configs/_base_/models/clipdinoiser.py
        self.model = MODELS.build(CLIP_DINOiser)(
            clip_backbone=dict(
                type="MaskClip",
                clip_model="ViT-B-16",
                backbone=dict(img_size=224, patch_size=16),
                decode_head=dict(
                    type="MaskClipHead",
                    in_channels=768,
                    text_channels=512,
                    pretrained="laion2b_s34b_b88k",
                    mode="compute_metric",
                    id_start=0,
                    id_end=79,
                ),
            ),
            class_names=self.class_names,
            data_preprocessor=dict(type="SegDataPreProcessor"),
            init_cfg=None,  # We'll load weights manually
            vit_arch="vit_base",
            vit_patch_size=16,
            enc_type_feats="v",
            gamma=0.2,
            delta=0.99,
            in_dim=256,
            feats_idx=-3,
        )

        # Load pretrained weights
        checkpoint = Path(PROJECT_ROOT) / self.checkpoint_path
        if checkpoint.exists():
            print(f"   Loading weights from {checkpoint}...")
            state_dict = torch.load(checkpoint, map_location="cpu")
            # Handle different checkpoint formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"   ⚠️ Checkpoint not found at {checkpoint}")

        self.model.eval().to(self.device)
        print(f"✅ CLIP-DINOiser loaded with {len(self.class_names)} classes!")

    @torch.no_grad()
    def compute_text_features(self) -> Dict[str, torch.Tensor]:
        """
        Compute text features using MaskClipHead's _embed_templates method.
        CLIP-DINOiser uses MaskCLIP's text encoder.

        Returns:
            Dictionary with per_template and averaged features
        """
        print(f"🔤 Computing text features for {len(self.class_names)} classes...")

        decode_head = self.model.clip_backbone.decode_head

        per_template_features = []
        for template_id in tqdm(range(80), desc="Computing text features"):
            # Use the existing _embed_templates method from MaskClipHead
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
        Compute refined vision features using CLIP-DINOiser's extract_feat.

        Args:
            image: Image tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Refined dense features [B, text_dim, h, w]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Use the existing extract_feat method which includes DINO refinement
        feat = self.model.extract_feat(image, template_dict=None)

        return feat.cpu()

    @property
    def model_name(self) -> str:
        return "clipdinoiser"
