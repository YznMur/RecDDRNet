#!/usr/bin/env python3
"""Compare eval results from multiple eval_metrics.json files.

Usage:
    python scripts/compare_results.py --dir output/
    python scripts/compare_results.py --dir output/ --csv results.csv --plots results/
    python scripts/compare_results.py path/to/model1/eval_metrics.json path/to/model2/eval_metrics.json --plots results/
"""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


CLASS_NAMES = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    label = os.path.basename(os.path.dirname(path))
    if label in ("", "."):
        label = os.path.basename(path)
    return label, data


def find_json_files(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f == "eval_metrics.json":
                results.append(os.path.join(root, f))
    return sorted(results)


def get_splits(results):
    all_splits = set()
    for _, data in results:
        all_splits.update(data.keys())
    preferred = ["test", "train", "val"]
    splits = [s for s in preferred if s in all_splits]
    splits += [s for s in sorted(all_splits) if s not in splits]
    return splits


def get_metric(results, split, key, default=None):
    vals = []
    for label, data in results:
        if split in data and key in data[split]:
            vals.append(data[split][key])
        else:
            vals.append(default)
    return vals


def print_comparison(results):
    if not results:
        print("No results to compare.")
        return

    labels = [r[0] for r in results]
    splits = get_splits(results)
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

        row = f"{'mIoU':<30s}"
        for v in get_metric(results, split, "mean_iou"):
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}s}"
        print(row)

        row = f"{'Loss':<30s}"
        for v in get_metric(results, split, "valid_loss"):
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}s}"
        print(row)

        row = f"{'Pixel Accuracy':<30s}"
        for v in get_metric(results, split, "pixel_acc"):
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}s}"
        print(row)

        print(f"\n{'Per-Class IoU':<30s}" + "-" * (col_w * len(labels)))
        for cls_name in CLASS_NAMES:
            row = f"  {cls_name:<28s}"
            for _, data in results:
                if split in data and "iou_per_class" in data[split]:
                    iou = data[split]["iou_per_class"].get(cls_name)
                    row += f"{iou:>{col_w}.4f}" if iou is not None else f"{'N/A':>{col_w}s}"
                else:
                    row += f"{'N/A':>{col_w}s}"
            print(row)

        has_kf = any(
            split in data and data[split].get("keyframe_interval", 1) > 1
            for _, data in results
        )
        if has_kf:
            row = f"\n{'Keyframe Interval':<30s}"
            for v in get_metric(results, split, "keyframe_interval"):
                row += f"{v:>{col_w}d}" if v is not None else f"{'N/A':>{col_w}s}"
            print(row)

        row = f"{'Time (s)':<30s}"
        for v in get_metric(results, split, "time_seconds"):
            row += f"{v:>{col_w}.1f}" if v is not None else f"{'N/A':>{col_w}s}"
        print(row)
        print()

    print(sep)
    print("BEST MODEL PER SPLIT")
    print(sep)
    for split in splits:
        best_label, best_miou = None, -1
        for label, data in results:
            if split in data and "mean_iou" in data[split]:
                if data[split]["mean_iou"] > best_miou:
                    best_miou = data[split]["mean_iou"]
                    best_label = label
        if best_label:
            print(f"  {split:>6s}: {best_label} (mIoU={best_miou:.4f})")
        else:
            print(f"  {split:>6s}: No results")
    print()


def save_csv(results, csv_path):
    splits = get_splits(results)
    with open(csv_path, "w") as f:
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


def plot_comparison(results, output_dir):
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    labels = [r[0] for r in results]
    splits = get_splits(results)
    n_models = len(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 1)))

    # 1. mIoU bar chart per split
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    for i, split in enumerate(splits):
        ax = axes[0][i]
        mious = get_metric(results, split, "mean_iou", 0)
        bars = ax.bar(range(n_models), mious, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mIoU")
        ax.set_title(f"{split.upper()} - mIoU")
        ax.set_ylim(0, max(mious) * 1.15 if max(mious) > 0 else 1)
        for bar, v in zip(bars, mious):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "miou_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'miou_comparison.png')}")

    # 2. Per-class IoU grouped bar chart per split
    for split in splits:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(CLASS_NAMES))
        width = 0.8 / n_models
        for j, (label, data) in enumerate(results):
            ious = []
            for cls_name in CLASS_NAMES:
                if split in data and "iou_per_class" in data[split]:
                    v = data[split]["iou_per_class"].get(cls_name, 0)
                    ious.append(v if isinstance(v, (int, float)) else 0)
                else:
                    ious.append(0)
            offset = (j - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, ious, width, label=label, color=colors[j],
                         edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
        ax.set_ylabel("IoU")
        ax.set_title(f"{split.upper()} - Per-Class IoU")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        fname = f"per_class_iou_{split}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(output_dir, fname)}")

    # 3. Radar/spider chart for best split
    if splits:
        split = splits[0]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES), endpoint=False).tolist()
        angles += angles[:1]
        for j, (label, data) in enumerate(results):
            ious = []
            for cls_name in CLASS_NAMES:
                if split in data and "iou_per_class" in data[split]:
                    v = data[split]["iou_per_class"].get(cls_name, 0)
                    ious.append(v if isinstance(v, (int, float)) else 0)
                else:
                    ious.append(0)
            ious += ious[:1]
            ax.plot(angles, ious, "o-", linewidth=2, label=label, color=colors[j])
            ax.fill(angles, ious, alpha=0.1, color=colors[j])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"{split.upper()} - Class IoU Radar", pad=20)
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'radar_chart.png')}")

    # 4. Inference time comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    times = get_metric(results, splits[0] if splits else "test", "time_seconds", 0)
    bars = ax.barh(range(n_models), times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Inference Time ({splits[0].upper() if splits else 'test'})")
    for bar, v in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}s", ha="left", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'inference_time.png')}")


def main():
    parser = argparse.ArgumentParser(description="Compare eval results from multiple models")
    parser.add_argument("files", nargs="*", help="Paths to eval_metrics.json files")
    parser.add_argument("--dir", type=str, help="Directory to recursively search for eval_metrics.json")
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument("--plots", type=str, help="Directory to save comparison charts")
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

    if args.plots:
        plot_comparison(results, args.plots)


if __name__ == "__main__":
    main()
