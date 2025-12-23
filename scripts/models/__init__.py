"""
FLOSS Model Feature Extractors

Thin wrappers around the existing model implementations in mmseg.models.segmentors.
"""

from .clipdinoiser import CLIPDINOiserExtractor
from .naclip import NACLIPExtractor
from .maskclip import MaskCLIPExtractor

__all__ = [
    "CLIPDINOiserExtractor",
    "NACLIPExtractor",
    "MaskCLIPExtractor",
    "get_extractor",
]


def get_extractor(model_name: str, **kwargs):
    """
    Factory function to get the appropriate feature extractor.

    Args:
        model_name: One of "clipdinoiser", "naclip", "maskclip"
        **kwargs: Additional arguments passed to the extractor

    Returns:
        Feature extractor instance
    """
    extractors = {
        "clipdinoiser": CLIPDINOiserExtractor,
        "naclip": NACLIPExtractor,
        "maskclip": MaskCLIPExtractor,
    }

    if model_name.lower() not in extractors:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(extractors.keys())}"
        )

    return extractors[model_name.lower()](**kwargs)
