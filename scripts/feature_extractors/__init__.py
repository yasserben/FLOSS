"""
FLOSS Feature Extractors

This module provides feature extractors for different OVSS models:
- CLIP-DINOiser
- MaskCLIP  
- NACLIP

Each extractor can compute text and vision representations using the model's
specific architecture and pretrained weights.
"""

from .base import BaseFeatureExtractor, IMAGENET_TEMPLATES, DATASET_CONFIGS
from .clipdinoiser import CLIPDINOiserExtractor
from .maskclip import MaskCLIPExtractor
from .naclip import NACLIPExtractor

__all__ = [
    "BaseFeatureExtractor",
    "CLIPDINOiserExtractor", 
    "MaskCLIPExtractor",
    "NACLIPExtractor",
    "IMAGENET_TEMPLATES",
    "DATASET_CONFIGS",
    "get_extractor",
]


def get_extractor(model_name: str, **kwargs):
    """
    Factory function to get the appropriate feature extractor.
    
    Args:
        model_name: One of 'clipdinoiser', 'maskclip', 'naclip'
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Feature extractor instance
    """
    extractors = {
        "clipdinoiser": CLIPDINOiserExtractor,
        "maskclip": MaskCLIPExtractor,
        "naclip": NACLIPExtractor,
    }
    
    model_name = model_name.lower()
    if model_name not in extractors:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(extractors.keys())}")
    
    return extractors[model_name](**kwargs)

