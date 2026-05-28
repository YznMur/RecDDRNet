#!/usr/bin/env python3
import cv2
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from natsort import natsorted  # for natural sorting

# =========================
# CONFIG
# =========================
DDRNET_ROOT = "./output_rsm_v2/onnx_output_results_test_rsm_v2"  
DATASET_ROOT = "./data/cityscapes/overlays/test"
OUTPUT_DIR = "./videos_onnx_rsmv2"

FPS = 75
TARGET_SIZE = (640, 480)

XLSX_MAP = {
    "train": "cvat_projects_train.xlsx",
    "val":   "cvat_projects_val.xlsx",
    "test":  "cvat_projects_test.xlsx",
}

# =========================
# ARGPARSE
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True, choices=["train", "val", "test"])
parser.add_argument("--max-print", type=int, default=20,
                    help="Limit printed paths (avoid huge spam)")
args = parser.parse_args()

SPLIT = args.split
MAX_PRINT = args.max_print

# =========================
# Utilities
# =========================
def resize_keep_aspect(img, target_size):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def extract_job_id(base_name):
    match = re.search(r"job_(\d+)", base_name)
    return int(match.group(1)) if match else None

def draw_label(img, text, pos=(10, 30), font_scale=0.5, thickness=1):
    """Draw white text on black rectangle background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = pos
    # Draw black rectangle
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), (0, 0, 0), -1)
    # Put white text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def get_overlay_path(ddrnet_fname, overlay_root):
    """Map DDRNet filename to overlay filename based on your pattern."""
    # Remove 'overlay_' prefix
    base = ddrnet_fname
    if base.startswith("overlay_"):
        base = base[len("overlay_"):]
    # Replace DDRNet suffix with overlay suffix
    base = base.replace("_leftImg8bit.png", "_overlay.png")
    overlay_path = os.path.join(overlay_root, base)
    return overlay_path

# =========================
# XLSX loader
# =========================
def load_job_project_map(xlsx_path):
    if not os.path.exists(xlsx_path):
        print(f"⚠️ XLSX not found: {xlsx_path}. Using unknown labels.")
        return {}
    df = pd.read_excel(xlsx_path)
    job_map = {}
    for _, row in df.iterrows():
        try:
            job_id = int(row["Job ID"])
            job_map[job_id] = {
                "project": str(row["Project Name"]),
                "task": str(row["Task Name"]),
            }
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")
            continue
    return job_map

# =========================
# Core video creation
# =========================
def create_side_by_side_video_from_folder(
    ddrnet_root,
    overlay_root,
    output_video,
    job_project_map,
    split_name,
    fps,
    target_size,
):
    ddrnet_files = [f for f in os.listdir(ddrnet_root) if f.endswith("_leftImg8bit.png")]
    ddrnet_files = natsorted(ddrnet_files)

    video = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_size[0] * 2, target_size[1]),
    )

    written = 0
    skipped = 0
    printed = 0

    for fname in tqdm(ddrnet_files, desc=f"{split_name.upper()}"):
        ddrnet_path = os.path.join(ddrnet_root, fname)
        overlay_path = get_overlay_path(fname, overlay_root)

        if printed < MAX_PRINT:
            print("DDRNet:", ddrnet_path)
            print("Overlay:", overlay_path)
            printed += 1

        ddrnet_img = cv2.imread(ddrnet_path)
        overlay_img = cv2.imread(overlay_path)

        if ddrnet_img is None:
            print("❌ MISSING DDRNet:", ddrnet_path)
            skipped += 1
            continue
        if overlay_img is None:
            print("❌ MISSING Overlay:", overlay_path)
            skipped += 1
            continue

        base_name_no_ext = os.path.splitext(fname)[0]
        job_id = extract_job_id(base_name_no_ext)
        meta = job_project_map.get(job_id, {})
        label = (
            f"{meta.get('project', 'UNKNOWN')} | "
            f"{meta.get('task', 'UNKNOWN')} | "
            f"{split_name.upper()} | {base_name_no_ext}"
        )

        left = resize_keep_aspect(ddrnet_img, target_size)
        right = resize_keep_aspect(overlay_img, target_size)

        # Draw text with black background
        draw_label(left, "DDRNet Prediction", pos=(10, 25), font_scale=0.3)
        draw_label(right, "Dataset Overlay", pos=(10, 25), font_scale=0.3)
        draw_label(left, label, pos=(10, 45), font_scale=0.3)
        draw_label(right, label, pos=(10, 45), font_scale=0.3)

        video.write(cv2.hconcat([left, right]))
        written += 1

    video.release()
    print("\n================ SUMMARY ================")
    print("Frames written :", written)
    print("Frames skipped :", skipped)
    print("========================================")
    return written

# =========================
# Main
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

xlsx_path = os.path.join(".", XLSX_MAP[SPLIT])
job_project_map = load_job_project_map(xlsx_path)

output_video = os.path.join(
    OUTPUT_DIR, f"{SPLIT}_ddrnet_vs_dataset_overlay.mp4"
)

frames = create_side_by_side_video_from_folder(
    ddrnet_root=DDRNET_ROOT,
    overlay_root=DATASET_ROOT,
    output_video=output_video,
    job_project_map=job_project_map,
    split_name=SPLIT,
    fps=FPS,
    target_size=TARGET_SIZE,
)

print(f"\n✅ {SPLIT.upper()} video created | frames: {frames}")
print("🎉 Done!")
