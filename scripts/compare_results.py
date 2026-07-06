#!/usr/bin/env python3
"""Compare eval results and generate a PDF report with charts and tables.

Usage:
    python scripts/compare_results.py --dir output/ --pdf report.pdf
    python scripts/compare_results.py path/to/model1/eval_metrics.json path/to/model2/eval_metrics.json --pdf report.pdf
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


CLASS_NAMES = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]
CMAP = plt.cm.Set2


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
    return [data[split].get(key, default) if split in data else default for _, data in results]


def truncate_label(label, max_len=25):
    return label if len(label) <= max_len else "..." + label[-(max_len - 3):]


def generate_pdf(results, pdf_path):
    labels = [r[0] for r in results]
    short_labels = [truncate_label(l) for l in labels]
    splits = get_splits(results)
    n_models = len(labels)
    colors = CMAP(np.linspace(0, 1, max(n_models, 1)))

    with PdfPages(pdf_path) as pdf:

        # ===== PAGE 1: TITLE =====
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        plt.text(0.5, 0.65, "DDRNet Model Comparison Report", fontsize=28,
                 fontweight="bold", ha="center", va="center", transform=fig.transFigure)
        plt.text(0.5, 0.52, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=12, ha="center", va="center", color="gray", transform=fig.transFigure)
        plt.text(0.5, 0.42, f"Models compared: {n_models}", fontsize=14,
                 ha="center", va="center", transform=fig.transFigure)
        model_list = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(labels))
        plt.text(0.5, 0.28, model_list, fontsize=11, ha="center", va="center",
                 family="monospace", transform=fig.transFigure)
        pdf.savefig(fig)
        plt.close()

        # ===== PAGE 2: SUMMARY TABLE =====
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("Summary - mIoU per Split", fontsize=16, fontweight="bold", pad=20)

        n_rows = len(splits)
        n_cols = n_models + 1
        table_data = []
        for split in splits:
            row = [split.upper()]
            for v in get_metric(results, split, "mean_iou"):
                row.append(f"{v:.4f}" if v is not None else "N/A")
            table_data.append(row)

        table = ax.table(cellText=table_data,
                         colLabels=["Split"] + short_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == 0:
                cell.set_facecolor("#D6E4F0")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")

        # Best model per split
        best_text = "Best model per split:\n"
        for split in splits:
            best_label, best_miou = None, -1
            for label, data in results:
                if split in data and "mean_iou" in data[split]:
                    if data[split]["mean_iou"] > best_miou:
                        best_miou = data[split]["mean_iou"]
                        best_label = label
            if best_label:
                best_text += f"  {split:>6s}: {truncate_label(best_label, 30)} (mIoU={best_miou:.4f})\n"
        plt.text(0.5, 0.08, best_text, fontsize=10, ha="center", va="center",
                 family="monospace", transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F4FD", edgecolor="#4472C4"))

        pdf.savefig(fig)
        plt.close()

        # ===== PAGES 3+: mIoU BAR CHART =====
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        fig.suptitle("mIoU Comparison", fontsize=16, fontweight="bold", y=0.95)
        gs = gridspec.GridSpec(1, len(splits), figure=fig, wspace=0.35)
        for i, split in enumerate(splits):
            ax = fig.add_subplot(gs[0, i])
            mious = get_metric(results, split, "mean_iou", 0)
            bars = ax.bar(range(n_models), mious, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("mIoU")
            ax.set_title(split.upper(), fontsize=12, fontweight="bold")
            ymax = max(mious) if mious and max(mious) > 0 else 1
            ax.set_ylim(0, ymax * 1.2)
            for bar, v in zip(bars, mious):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        pdf.savefig(fig)
        plt.close()

        # ===== PAGES: PER-CLASS IOU GROUPED BARS =====
        for split in splits:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.set_title(f"{split.upper()} - Per-Class IoU", fontsize=16, fontweight="bold")
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
                bars = ax.bar(x + offset, ious, width, label=truncate_label(label, 30),
                              color=colors[j], edgecolor="black", linewidth=0.3)
                for bar, v in zip(bars, ious):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_NAMES, fontsize=10)
            ax.set_ylabel("IoU")
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # ===== PAGE: RADAR CHART =====
        for split in splits:
            fig = plt.figure(figsize=(8.27, 8.27))
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111, polar=True)
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
                ax.plot(angles, ious, "o-", linewidth=2, label=truncate_label(label, 30), color=colors[j])
                ax.fill(angles, ious, alpha=0.08, color=colors[j])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(CLASS_NAMES, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(f"{split.upper()} - Class IoU Radar", fontsize=14, fontweight="bold", pad=25)
            ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.1))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # ===== PAGE: INFERENCE TIME =====
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_title(f"Inference Time ({splits[0].upper() if splits else 'TEST'})",
                     fontsize=16, fontweight="bold")
        split = splits[0] if splits else "test"
        times = get_metric(results, split, "time_seconds", 0)
        bars = ax.barh(range(n_models), times, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("Time (seconds)")
        for bar, v in zip(bars, times):
            ax.text(bar.get_width() + max(times) * 0.01 if max(times) > 0 else 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}s", ha="left", va="center", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ===== PAGE: DETAILED TABLE PER SPLIT =====
        for split in splits:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(f"{split.upper()} - Detailed Results", fontsize=16,
                         fontweight="bold", pad=20)

            header_row = ["Metric"] + short_labels
            table_data = []

            # mIoU
            row = ["mIoU"]
            for v in get_metric(results, split, "mean_iou"):
                row.append(f"{v:.4f}" if v is not None else "N/A")
            table_data.append(row)

            # Loss
            row = ["Loss"]
            for v in get_metric(results, split, "valid_loss"):
                row.append(f"{v:.4f}" if v is not None else "N/A")
            table_data.append(row)

            # Pixel Acc
            row = ["Pixel Acc"]
            for v in get_metric(results, split, "pixel_acc"):
                row.append(f"{v:.4f}" if v is not None else "N/A")
            table_data.append(row)

            # Per-class IoU
            for cls_name in CLASS_NAMES:
                row = [f"  {cls_name}"]
                for _, data in results:
                    if split in data and "iou_per_class" in data[split]:
                        v = data[split]["iou_per_class"].get(cls_name)
                        row.append(f"{v:.4f}" if v is not None else "N/A")
                    else:
                        row.append("N/A")
                table_data.append(row)

            # Time
            row = ["Time (s)"]
            for v in get_metric(results, split, "time_seconds"):
                row.append(f"{v:.1f}" if v is not None else "N/A")
            table_data.append(row)

            # Keyframe
            has_kf = any(
                split in data and data[split].get("keyframe_interval", 1) > 1
                for _, data in results
            )
            if has_kf:
                row = ["Keyframe Interval"]
                for v in get_metric(results, split, "keyframe_interval"):
                    row.append(str(v) if v is not None else "N/A")
                table_data.append(row)

                row = ["Backbone Runs"]
                for v in get_metric(results, split, "backbone_runs"):
                    row.append(str(v) if v is not None else "N/A")
                table_data.append(row)

            table = ax.table(cellText=table_data, colLabels=header_row,
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.6)

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#4472C4")
                    cell.set_text_props(color="white", fontweight="bold")
                elif col == 0:
                    cell.set_facecolor("#D6E4F0")
                    cell.set_text_props(fontweight="bold")
                else:
                    cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")
                    # Highlight max mIoU
                    if row == 1 and col > 0:
                        try:
                            val = float(cell.get_text().get_text())
                            max_val = max(float(t.get_text().get_text())
                                         for r2, t in table.get_celld().items()
                                         if r2 == 1 and r2 == row and col > 0
                                         and t.get_text().get_text() not in ("N/A", ""))
                            if val == max_val:
                                cell.set_facecolor("#92D050")
                                cell.set_text_props(fontweight="bold")
                        except (ValueError, AttributeError):
                            pass

            pdf.savefig(fig)
            plt.close()

    print(f"PDF report saved to: {pdf_path}")


def print_comparison(results):
    if not results:
        return
    labels = [r[0] for r in results]
    splits = get_splits(results)
    col_w = max(22, max(len(l) for l in labels) + 2)
    header = f"{'Metric':<30s}" + "".join(f"{l:>{col_w}s}" for l in labels)
    sep = "=" * len(header)

    print(sep)
    print("COMPARISON REPORT")
    print(sep)
    for split in splits:
        print(f"\n--- {split.upper()} ---")
        print(header)
        print("-" * len(header))
        for metric_name, metric_key in [("mIoU", "mean_iou"), ("Loss", "valid_loss"),
                                         ("Pixel Accuracy", "pixel_acc")]:
            row = f"{metric_name:<30s}"
            for v in get_metric(results, split, metric_key):
                row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}s}"
            print(row)
        for cls_name in CLASS_NAMES:
            row = f"  {cls_name:<28s}"
            for _, data in results:
                if split in data and "iou_per_class" in data[split]:
                    v = data[split]["iou_per_class"].get(cls_name)
                    row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}s}"
                else:
                    row += f"{'N/A':>{col_w}s}"
            print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare eval results - generate PDF report")
    parser.add_argument("files", nargs="*", help="Paths to eval_metrics.json files")
    parser.add_argument("--dir", type=str, help="Directory to recursively search for eval_metrics.json")
    parser.add_argument("--pdf", type=str, default="comparison_report.pdf",
                        help="Output PDF path (default: comparison_report.pdf)")
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
    generate_pdf(results, args.pdf)


if __name__ == "__main__":
    main()
