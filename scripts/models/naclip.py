"""
NACLIP Feature Extractor - Wrapper around existing implementation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import mmseg.models to register all components (including SegDataPreProcessor)
import mmseg.models  # This registers SegDataPreProcessor in mmengine registry

# Import the existing NACLIP implementation
from mmseg.models.segmentors.naclip import NACLIP
from mmseg.models.segmentors.naclip.clip import load, tokenize
from mmseg.models.segmentors.naclip.prompts.imagenet_template import (
    openai_imagenet_template,
)


class NACLIPExtractor:
    """
    Feature extractor wrapper for NACLIP.
    Uses the existing NACLIP class directly.
    """

    def __init__(
        self,
        class_names: List[str],
        name_path: str,
        device: str = "cuda",
        clip_path: str = "ViT-B/16",
    ):
        """
        Args:
            class_names: List of class names
            name_path: Path to the class names file (e.g., configs/naclip/cls_cityscapes.txt)
            device: Device to use
            clip_path: CLIP model path
        """
        self.class_names = class_names
        self.name_path = name_path
        self.device = device
        self.clip_path = clip_path
        self.templates = [t("{}") for t in openai_imagenet_template]  # Convert lambdas
        self.model = None
        self.net = None  # Direct CLIP model reference
        self._load_model()

    def _load_model(self):
        """Load NACLIP model."""
        print(f"📥 Loading NACLIP ({self.clip_path})...")

        # Load CLIP model directly for feature extraction
        self.net, _ = load(self.clip_path, device=self.device, jit=False)

        # Also create NACLIP model for full functionality
        self.model = NACLIP(
            clip_path=self.clip_path,
            name_path=self.name_path,
            device=torch.device(self.device),
            mode="compute_metric",
            id_start=0,
            id_end=79,
        )

        print(f"✅ NACLIP loaded with {len(self.class_names)} classes!")

    @torch.no_grad()
    def compute_text_features(self) -> Dict[str, torch.Tensor]:
        """
        Compute text features for all templates.

        Returns:
            Dictionary with per_template and averaged features
        """
        print(f"🔤 Computing text features for {len(self.class_names)} classes...")

        per_template_features = []

        for template_id in tqdm(range(80), desc="Computing text features"):
            template_fn = openai_imagenet_template[template_id]
            template_features = []

            for class_name in self.class_names:
                query = tokenize([template_fn(class_name)]).to(self.device)
                feature = self.net.encode_text(query)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature = feature / feature.norm()
                template_features.append(feature.cpu())

            per_template_features.append(torch.stack(template_features))

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
        Compute vision features using NACLIP's encode_image.

        Args:
            image: Image tensor [B, C, H, W]

        Returns:
            Dense features [B, dim, H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Use NACLIP's visual encoder
        image_features = self.net.encode_image(image, return_all=True)
        image_features = image_features[:, 1:]  # Remove CLS token
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Reshape to spatial format
        B, N, C = image_features.shape
        patch_size = self.net.visual.patch_size
        h = image.shape[-2] // patch_size
        w = image.shape[-1] // patch_size
        image_features = image_features.permute(0, 2, 1).reshape(B, C, h, w)

        return image_features.cpu()

    @property
    def model_name(self) -> str:
        return "naclip"
