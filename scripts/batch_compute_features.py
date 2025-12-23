#!/usr/bin/env python3
"""
Batch compute FLOSS features for all datasets.

This script automates the computation of text and vision features
for all supported datasets, ready for HuggingFace upload.

Usage:
    python scripts/batch_compute_features.py --compute-text
    python scripts/batch_compute_features.py --compute-text --upload --hf-repo username/repo
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASETS = [
    "cityscapes",
    "pascalvoc20",
    "pascalco59",
    "ade20k",
    "cocostuff",
]


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"❌ Error running: {' '.join(cmd)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch compute FLOSS features")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to process",
    )
    parser.add_argument(
        "--compute-text", action="store_true", help="Compute text features"
    )
    parser.add_argument(
        "--compute-vision",
        action="store_true",
        help="Compute vision features (warning: very large)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./floss_features", help="Output directory"
    )
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repository ID")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")

    args = parser.parse_args()

    if not args.compute_text and not args.compute_vision:
        print("❌ Please specify at least one of: --compute-text, --compute-vision")
        sys.exit(1)

    if args.upload and not args.hf_repo:
        print("❌ --hf-repo required for upload")
        sys.exit(1)

    # Process each dataset
    results = {}

    for dataset in args.datasets:
        print(f"\n{'#'*60}")
        print(f"# Processing: {dataset}")
        print(f"{'#'*60}")

        cmd = [
            sys.executable,
            "scripts/compute_features_hf.py",
            "--dataset",
            dataset,
            "--output-dir",
            args.output_dir,
        ]

        if args.compute_text:
            cmd.append("--compute-text")
        if args.compute_vision:
            cmd.append("--compute-vision")
        if args.upload:
            cmd.extend(["--upload", "--hf-repo", args.hf_repo])
            if args.hf_token:
                cmd.extend(["--hf-token", args.hf_token])

        success = run_command(cmd, f"Computing features for {dataset}")
        results[dataset] = "✅" if success else "❌"

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    for dataset, status in results.items():
        print(f"   {status} {dataset}")

    # Upload combined if requested
    if args.upload and all(r == "✅" for r in results.values()):
        print("\n✅ All features computed and uploaded successfully!")
    elif args.upload:
        print("\n⚠️ Some datasets failed. Check the logs above.")
    else:
        print(f"\n✅ Features saved to: {args.output_dir}")
        print("   To upload, run with --upload --hf-repo your-username/repo")


if __name__ == "__main__":
    main()
