#!/usr/bin/env python3
"""
FLOSS: Feature Extraction Script for HuggingFace Dataset Upload

This script computes vision and text representations using FLOSS models:
- CLIP-DINOiser (default)
- NACLIP
- MaskCLIP

The computed features can be uploaded to HuggingFace datasets.

Author: FLOSS Team
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import extractors from local models module
from models import get_extractor

# Import templates
from mmseg.models.segmentors.maskclip.utils.prompt_templates import imagenet_templates

# HuggingFace imports
try:
    from huggingface_hub import HfApi, login

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Upload functionality disabled.")


# Dataset configurations
DATASET_CONFIGS = {
    "cityscapes": {
        "num_classes": 19,
        "class_names": [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
        "name_path": "configs/naclip/cls_cityscapes.txt",
    },
    "ade20k": {
        "num_classes": 150,
        "class_names": None,  # Will be loaded from file
        "name_path": "configs/naclip/cls_ade20k.txt",
    },
    "cocostuff": {
        "num_classes": 171,
        "class_names": None,
        "name_path": "configs/naclip/cls_coco_stuff.txt",
    },
    "pascalvoc20": {
        "num_classes": 20,
        "class_names": [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ],
        "name_path": "configs/naclip/cls_voc20.txt",
    },
    "pascalco59": {
        "num_classes": 59,
        "class_names": None,
        "name_path": "configs/naclip/cls_context59.txt",
    },
}


def load_class_names(dataset_name: str) -> List[str]:
    """Load class names from config files."""
    # First try DATASET_CONFIGS
    if DATASET_CONFIGS[dataset_name].get("class_names"):
        return DATASET_CONFIGS[dataset_name]["class_names"]

    # Try loading from class file
    class_file = (
        PROJECT_ROOT
        / "configs"
        / "_base_"
        / "datasets"
        / "class_names"
        / f"{dataset_name}_classes.py"
    )
    if class_file.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("classes", class_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.list_of_classes

    # Try loading from NACLIP cls file
    name_path = PROJECT_ROOT / DATASET_CONFIGS[dataset_name].get("name_path", "")
    if name_path.exists():
        with open(name_path, "r") as f:
            lines = f.read().strip().split("\n")
            # Parse class names (format: "classname" or "classname1,classname2")
            class_names = []
            for line in lines:
                parts = line.split(",")
                class_names.append(parts[0].strip())
            return class_names

    return []


def save_features_to_disk(
    features_dict: Dict, output_dir: Path, dataset_name: str, model_name: str
):
    """Save computed features to disk."""
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text features
    if "text_features" in features_dict:
        text_dir = output_dir / "text_features"
        text_dir.mkdir(exist_ok=True)
        torch.save(
            features_dict["text_features"],
            text_dir / f"{dataset_name}_text_features.pt",
        )
        print(f"  📁 Text features: {text_dir / f'{dataset_name}_text_features.pt'}")

    # Save vision features
    if "vision_features" in features_dict:
        vision_dir = output_dir / "vision_features"
        vision_dir.mkdir(exist_ok=True)
        torch.save(
            features_dict["vision_features"],
            vision_dir / f"{dataset_name}_vision_features.pt",
        )
        print(
            f"  📁 Vision features: {vision_dir / f'{dataset_name}_vision_features.pt'}"
        )

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "num_templates": len(imagenet_templates),
        "templates": imagenet_templates,
        "class_names": features_dict.get("class_names", []),
    }

    with open(output_dir / f"{dataset_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  📁 Metadata: {output_dir / f'{dataset_name}_metadata.json'}")


def upload_to_huggingface(
    features_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = False
):
    """Upload computed features to HuggingFace Hub."""
    if not HF_AVAILABLE:
        print(
            "HuggingFace Hub not available. Please install: pip install huggingface_hub"
        )
        return

    if token:
        login(token=token)

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=str(features_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"✅ Features uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute FLOSS features for HuggingFace upload"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process (default: cityscapes)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clipdinoiser",
        choices=["clipdinoiser", "naclip", "maskclip"],
        help="Model to use for feature extraction (default: clipdinoiser)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./floss_features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--compute-text",
        action="store_true",
        default=True,
        help="Compute text features (default: True)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip text feature computation",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., 'username/floss-features')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/clip_dinoiser/model_weights.pth",
        help="Path to checkpoint (for CLIP-DINOiser)",
    )

    args = parser.parse_args()

    # Handle --no-text flag
    compute_text = not args.no_text

    # Load class names
    class_names = load_class_names(args.dataset)
    if not class_names:
        print(f"❌ Error: Could not load class names for {args.dataset}")
        return

    print(f"\n🚀 FLOSS Feature Extraction")
    print(f"=" * 50)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Classes:  {len(class_names)}")
    print(f"  Device:   {args.device}")
    print(f"=" * 50)

    # Create extractor based on model choice
    print(f"\n📥 Loading {args.model.upper()} model...")

    if args.model == "clipdinoiser":
        extractor = get_extractor(
            "clipdinoiser",
            class_names=class_names,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    elif args.model == "naclip":
        name_path = str(PROJECT_ROOT / DATASET_CONFIGS[args.dataset]["name_path"])
        extractor = get_extractor(
            "naclip",
            class_names=class_names,
            name_path=name_path,
            device=args.device,
        )
    elif args.model == "maskclip":
        extractor = get_extractor(
            "maskclip",
            class_names=class_names,
            device=args.device,
        )
    else:
        print(f"❌ Unknown model: {args.model}")
        return

    features_dict = {
        "class_names": class_names,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute text features
    if compute_text:
        print(f"\n🔤 Computing text features...")
        text_features = extractor.compute_text_features()
        features_dict["text_features"] = text_features
        print(f"  ✅ Text features shape: {text_features['per_template'].shape}")
        print(f"     [num_templates, num_classes, dim]")

    # Save to disk
    print(f"\n💾 Saving features...")
    save_features_to_disk(features_dict, output_dir, args.dataset, args.model)

    # Upload to HuggingFace if requested
    if args.upload:
        if not args.hf_repo:
            print("❌ Error: --hf-repo required for upload")
            return
        print(f"\n☁️ Uploading to HuggingFace...")
        upload_to_huggingface(
            features_dir=output_dir,
            repo_id=args.hf_repo,
            token=args.hf_token,
        )

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
