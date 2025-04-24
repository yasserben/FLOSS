"""
This script performs sanity checks on the template rankings JSON files to ensure their validity.
It verifies that:
1. All template IDs are within the valid range (0-79)
2. Each template ID appears exactly the same number of times
3. No template IDs are missing or duplicated

Usage:
    python sanity_check.py path/to/rankings.json

The script is particularly useful when computing your own template rankings to ensure
they follow the expected format and constraints before using them with FLOSS.
"""

import argparse
import json
from collections import defaultdict


def check_template_ids(json_path):
    """
    Check if template_ids in the JSON file:
    1. Only contain values from 0 to 79
    2. Each template_id appears the same number of times
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        return False
    except json.JSONDecodeError:
        print(f"Error: File {json_path} is not a valid JSON file")
        return False

    # Counter for template_ids
    template_counts = defaultdict(int)

    # Collect all template_ids and their counts
    for class_data in data["classes"].values():
        if "entropy_ranking" in class_data:
            for entry in class_data["entropy_ranking"]:
                if "template_id" in entry:
                    template_counts[entry["template_id"]] += 1

    # Check if all template_ids are between 0 and 79
    valid_range = set(range(80))  # 0 to 79
    found_ids = set(template_counts.keys())

    # Check for invalid IDs (outside 0-79)
    invalid_ids = found_ids - valid_range
    if invalid_ids:
        print(f"Error: Found invalid template_ids: {sorted(invalid_ids)}")
        return False

    # Check for missing IDs
    missing_ids = valid_range - found_ids
    if missing_ids:
        print(f"Error: Missing template_ids: {sorted(missing_ids)}")
        return False

    # Check if all template_ids appear the same number of times
    counts = list(template_counts.values())
    if len(set(counts)) != 1:
        print("Error: Not all template_ids appear the same number of times")
        print("Template ID counts:")
        for template_id, count in sorted(template_counts.items()):
            print(f"Template {template_id}: {count} times")
        return False

    print(f"Success: All template_ids (0-79) appear exactly {counts[0]} times each")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check template IDs in JSON file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    check_template_ids(args.json_path)


if __name__ == "__main__":
    main()
