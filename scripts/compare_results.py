#!/usr/bin/env python3
"""Compare eval results from multiple eval_metrics.json files.

Usage:
    python scripts/compare_results.py path/to/eval1/eval_metrics.json path/to/eval2/eval_metrics.json ...
    python scripts/compare_results.py --dir path/to/output/  (finds all eval_metrics.json under subdirs)
    python scripts/compare_results.py --dir path/to/output/ --csv results.csv
"""

import argparse
import json
import os
import sys


CLASS_NAMES = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]


def load_results(path):
    """Load eval_metrics.json and return (label, data)."""
    with open(path) as f:
        data = json.load(f)
    # Use parent directory name as label
    label = os.path.basename(os.path.dirname(path))
    if label in ("", "."):
        label = os.path.basename(path)
    return label, data


def find_json_files(directory):
    """Recursively find all eval_metrics.json files under directory."""
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f == "eval_metrics.json":
                results.append(os.path.join(root, f))
    return sorted(results)


def print_comparison(results):
    """Print a formatted comparison table."""
    if not results:
        print("No results to compare.")
        return

    labels = [r[0] for r in results]
    datasets = ["test", "train", "val"]

    # Find which splits exist
    all_splits = set()
    for _, data in results:
        all_splits.update(data.keys())
    # Prefer test, train, val order
    splits = [s for s in datasets if s in all_splits]
    splits += [s for s in sorted(all_splits) if s not in splits]

    # Print summary table
    col_w = max(22, max(len(l) for l in labels) + 2)
    header = f"{'Metric':<30s}" + "".join(f"{l:>{col_w}s}" for l in labels)
    sep = "=" * len(header)

    print(sep)
    print("COMPARISON REPORT")
    print(sep)
    print()

    for split in splits:
        print(f"--- {split.upper()} ---")
        print(header)
        print("-" * len(header))

        # mIoU
        row = f"{'mIoU':<30s}"
        for _, data in results:
            if split in data and "mean_iou" in data[split]:
                row += f"{data[split]['mean_iou']:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}s}"
        print(row)

        # Loss
        row = f"{'Loss':<30s}"
        for _, data in results:
            if split in data and "valid_loss" in data[split]:
                row += f"{data[split]['valid_loss']:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}s}"
        print(row)

        # Pixel accuracy
        row = f"{'Pixel Accuracy':<30s}"
        for _, data in results:
            if split in data and "pixel_acc" in data[split]:
                row += f"{data[split]['pixel_acc']:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}s}"
        print(row)

        # Per-class IoU
        print(f"\n{'Per-Class IoU':<30s}" + "-" * (col_w * len(labels)))
        for cls_name in CLASS_NAMES:
            row = f"  {cls_name:<28s}"
            for _, data in results:
                if split in data and "iou_per_class" in data[split]:
                    iou = data[split]["iou_per_class"].get(cls_name, None)
                    if iou is not None:
                        row += f"{iou:>{col_w}.4f}"
                    else:
                        row += f"{'N/A':>{col_w}s}"
                else:
                    row += f"{'N/A':>{col_w}s}"
            print(row)

        # Keyframe info (if present)
        has_kf = any(
            split in data and data[split].get("keyframe_interval", 1) > 1
            for _, data in results
        )
        if has_kf:
            row = f"\n{'Keyframe Interval':<30s}"
            for _, data in results:
                if split in data and "keyframe_interval" in data[split]:
                    row += f"{data[split]['keyframe_interval']:>{col_w}d}"
                else:
                    row += f"{'N/A':>{col_w}s}"
            print(row)

            row = f"{'Backbone Runs':<30s}"
            for _, data in results:
                if split in data and "backbone_runs" in data[split]:
                    row += f"{data[split]['backbone_runs']:>{col_w}}"
                else:
                    row += f"{'N/A':>{col_w}s}"
            print(row)

        # Time
        row = f"{'Time (s)':<30s}"
        for _, data in results:
            if split in data and "time_seconds" in data[split]:
                row += f"{data[split]['time_seconds']:>{col_w}.1f}"
            else:
                row += f"{'N/A':>{col_w}s}"
        print(row)

        print()

    # Find best model per split
    print(sep)
    print("BEST MODEL PER SPLIT")
    print(sep)
    for split in splits:
        best_label = None
        best_miou = -1
        for label, data in results:
            if split in data and "mean_iou" in data[split]:
                miou = data[split]["mean_iou"]
                if miou > best_miou:
                    best_miou = miou
                    best_label = label
        if best_label:
            print(f"  {split:>6s}: {best_label} (mIoU={best_miou:.4f})")
        else:
            print(f"  {split:>6s}: No results")
    print()


def save_csv(results, csv_path):
    """Save comparison to CSV."""
    all_splits = set()
    for _, data in results:
        all_splits.update(data.keys())
    datasets = ["test", "train", "val"]
    splits = [s for s in datasets if s in all_splits]
    splits += [s for s in sorted(all_splits) if s not in splits]

    with open(csv_path, "w") as f:
        # Header
        cols = ["split", "model"] + CLASS_NAMES + ["mIoU", "loss", "pixel_acc", "time_s"]
        f.write(",".join(cols) + "\n")

        for split in splits:
            for label, data in results:
                if split not in data or "mean_iou" not in data[split]:
                    continue
                row = [split, label]
                iou_per_class = data[split].get("iou_per_class", {})
                for cls_name in CLASS_NAMES:
                    v = iou_per_class.get(cls_name, "")
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
                row.append(f"{data[split]['mean_iou']:.4f}")
                row.append(f"{data[split].get('valid_loss', ''):.4f}" if "valid_loss" in data[split] else "")
                row.append(f"{data[split].get('pixel_acc', ''):.4f}" if "pixel_acc" in data[split] else "")
                row.append(f"{data[split].get('time_seconds', ''):.1f}" if "time_seconds" in data[split] else "")
                f.write(",".join(str(v) for v in row) + "\n")

    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare eval results from multiple models")
    parser.add_argument("files", nargs="*", help="Paths to eval_metrics.json files")
    parser.add_argument("--dir", type=str, help="Directory to recursively search for eval_metrics.json")
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    json_files = list(args.files)

    if args.dir:
        json_files.extend(find_json_files(args.dir))

    if not json_files:
        print("No files provided. Use --dir or pass file paths.")
        sys.exit(1)

    results = []
    for path in json_files:
        try:
            label, data = load_results(path)
            results.append((label, data))
            print(f"Loaded: {path} -> {label}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not results:
        print("No valid results loaded.")
        sys.exit(1)

    print()
    print_comparison(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
