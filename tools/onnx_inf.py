#!/usr/bin/env python3
import os
import cv2
import numpy as np
import onnxruntime
import argparse
import time
import glob
import csv
from collections import defaultdict

# -----------------------------
# Utilities
# -----------------------------
def preprocess_image(img, target_size):
    H, W = target_size
    img_resized = cv2.resize(img, (W, H))
    img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize 0-1
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    img_chw = img_norm.transpose(2, 0, 1)  # HWC -> CHW
    img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
    return img_batch

def get_colormap(n_classes):
    # Distinct colors
    colormap = [
        [255,255,0],      # 0 yellow
        [255,0,255],      # 1 magenta
        [0,255,255],      # 2 cyan
        [128,128,0],      # 3 olive
        [128,0,128],      # fallback
        [0,128,128],
        [128,128,128],
        [255,128,0],
        [128,0,255],
        [0,128,255],
        [128,255,0],
        [255,0,128],
        [0,255,128],
        [255,255,255]
    ]
    if n_classes > len(colormap):
        colormap = (colormap * ((n_classes // len(colormap)) + 1))[:n_classes]
    return np.array(colormap, dtype=np.uint8)

def mask_to_color(mask, colormap):
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        color_mask[mask == i] = color
    return color_mask

def overlay_mask_on_image(image, mask, alpha=0.6):
    # image and mask are BGR uint8
    overlay = cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0)
    return overlay

def find_gt_mask_for_image(img_path, gt_root):
    """
    Try to locate a matching GT mask under gt_root by substring matching.
    Returns path or None.
    """
    base = os.path.basename(img_path)
    name_no_ext = os.path.splitext(base)[0]
    # Cityscapes image file name contains '_leftImg8bit' typically.
    if name_no_ext.endswith("_leftImg8bit"):
        prefix = name_no_ext.replace("_leftImg8bit", "")
    else:
        prefix = name_no_ext

    # Search recursively for a file that contains prefix and ends with '_labelIds.png' or 'labelIds.png'
    pattern1 = os.path.join(gt_root, "**", f"{prefix}*_labelIds.png")
    pattern2 = os.path.join(gt_root, "**", f"{prefix}*labelIds.png")
    matches = glob.glob(pattern1, recursive=True) + glob.glob(pattern2, recursive=True)
    if len(matches) > 0:
        # pick first, ideally there's only one
        return matches[0]
    # fallback: search any mask that contains the prefix as substring
    pattern3 = os.path.join(gt_root, "**", f"*{prefix}*.png")
    matches2 = glob.glob(pattern3, recursive=True)
    for m in matches2:
        if "label" in os.path.basename(m).lower() or "gt" in os.path.basename(m).lower():
            return m
    return None

# -----------------------------
# IoU utilities
# -----------------------------
def update_confusion(pred_mask, gt_mask, num_classes, ignore_label=255):
    """
    Returns arrays intersections and unions per class to be accumulated across dataset.
    pred_mask, gt_mask must be same shape.
    """

    # flatten
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    intersections = np.zeros(num_classes, dtype=np.uint64)
    unions = np.zeros(num_classes, dtype=np.uint64)
    # for each class
    for c in range(num_classes):
        gt_c = (gt_flat == c)
        pred_c = (pred_flat == c)
        # ignore where gt == ignore_label
        valid = (gt_flat != ignore_label)
        inter = np.logical_and(gt_c, pred_c) & valid
        union = np.logical_or(gt_c, pred_c) & valid
        intersections[c] = inter.sum()
        unions[c] = union.sum()
    return intersections, unions

def compute_iou_from_accum(intersections, unions):
    num_classes = len(intersections)
    ious = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if unions[c] > 0:
            ious[c] = intersections[c] / unions[c]
        else:
            ious[c] = float('nan')  # no pixels for this class in GT and predictions
    # mIoU: mean of classes that have union>0
    valid_mask = ~np.isnan(ious)
    if valid_mask.sum() > 0:
        miou = np.nanmean(ious)
    else:
        miou = float('nan')
    return ious, miou

# -----------------------------
# Inference
# -----------------------------
def infer_onnx(model_path,
               image_path,
               output_dir="output",
               gt_root=None,
               split="val",
               num_classes=4,
               input_size=(512,1024),
               alpha=0.6,
               device_providers=['CPUExecutionProvider']):
    os.makedirs(output_dir, exist_ok=True)
    session = onnxruntime.InferenceSession(model_path, providers=device_providers)

    # Input / output names & shapes
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape  # e.g. [1,3,512,1024] or [None,3,None,None]
    # try to fetch H,W from model shape; fallback to provided input_size
    try:
        H_in = int(input_shape[2]) if input_shape[2] is not None else input_size[0]
        W_in = int(input_shape[3]) if input_shape[3] is not None else input_size[1]
    except Exception:
        H_in, W_in = input_size

    out_meta = session.get_outputs()[0]
    output_name = out_meta.name
    # Try to deduce number of classes from output shape, else use provided
    try:
        out_shape = out_meta.shape  # e.g. [1, C, H, W]
        if out_shape[1] is None:
            n_classes = num_classes
        else:
            n_classes = int(out_shape[1])
    except Exception:
        n_classes = num_classes

    if n_classes != num_classes:
        print(f"Warning: model output channels ({n_classes}) != provided num_classes ({num_classes}). Using model value.")
        num_classes = n_classes

    colormap = get_colormap(num_classes)

    # gather image files
    if os.path.isdir(image_path):
        img_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    else:
        img_files = [image_path]

    # Prepare accumulators
    total_infer_time = 0.0
    total_total_time = 0.0
    infer_count = 0

    accum_intersections = np.zeros(num_classes, dtype=np.uint64)
    accum_unions = np.zeros(num_classes, dtype=np.uint64)

    # CSV summary
    csv_rows = []
    csv_header = ["image", "has_gt", "per_class_iou", "mIoU", "inference_time_s", "total_time_s"]

    for img_file in img_files:
        img_orig = cv2.imread(img_file)
        if img_orig is None:
            print(f"Failed to read image: {img_file}. Skipping.")
            continue
        H_orig, W_orig = img_orig.shape[:2]

        # find matching GT mask if gt_root provided
        gt_path = None
        if gt_root:
            # allow user to pass gt_root pointing to root (containing train/val/test) OR directly to split
            if os.path.isdir(gt_root) and os.path.basename(gt_root) in ("train", "val", "test"):
                gt_search_root = gt_root
            else:
                gt_search_root = os.path.join(gt_root, split)
            if os.path.isdir(gt_search_root):
                gt_path = find_gt_mask_for_image(img_file, gt_search_root)
                if gt_path is None:
                    # no match found
                    pass

        # Preprocess + inference + postprocess timing
        t0 = time.perf_counter()
        x = preprocess_image(img_orig, target_size=(H_in, W_in))
        t_pre = time.perf_counter()

        t_inf_start = time.perf_counter()
        pred = session.run(None, {input_name: x})[0]  # expected (1, C, H, W)
        t_inf_end = time.perf_counter()

        # postprocess
        # handle shape (1,C,H,W) -> take argmax over channel axis
        if pred.ndim == 4:
            pred_mask_small = np.argmax(pred[0], axis=0).astype(np.uint8)
        elif pred.ndim == 3:
            # sometimes output might be (C,H,W) already
            pred_mask_small = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

        # Resize predicted mask to original image size
        pred_mask_full = cv2.resize(pred_mask_small, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        t_post = time.perf_counter()
        total_time = t_post - t0
        infer_time = t_inf_end - t_inf_start

        total_total_time += total_time
        total_infer_time += infer_time
        infer_count += 1

        # Save mask and overlay
        color_mask = mask_to_color(pred_mask_full, colormap)
        overlay = overlay_mask_on_image(img_orig, color_mask, alpha=alpha)

        base_name = os.path.basename(img_file)
        mask_out_path = os.path.join(output_dir, f"mask_{base_name}")
        overlay_out_path = os.path.join(output_dir, f"overlay_{base_name}")
        cv2.imwrite(mask_out_path, color_mask[:, :, ::-1])      # convert RGB->BGR for saving
        cv2.imwrite(overlay_out_path, overlay[:, :, ::-1])

        # If GT exists, load and compute per-image IoU
        per_img_iou = None
        per_img_miou = None
        if gt_path and os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_mask is None:
                print(f"Warning: could not read GT mask {gt_path}")
            else:
                # If GT mask has multiple channels, take first
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask[:,:,0]
                # Resize GT to original image size if needed (usually GT is original size already)
                if gt_mask.shape[:2] != (H_orig, W_orig):
                    gt_mask = cv2.resize(gt_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

                intersections, unions = update_confusion(pred_mask_full, gt_mask, num_classes=num_classes, ignore_label=255)
                accum_intersections += intersections
                accum_unions += unions

                ious, miou = compute_iou_from_accum(intersections, unions)
                per_img_iou = ";".join([f"{x:.4f}" if not np.isnan(x) else "nan" for x in ious.tolist()])
                per_img_miou = f"{miou:.4f}" if not np.isnan(miou) else "nan"
        else:
            # No GT found
            pass

        csv_rows.append([base_name,
                         bool(gt_path and os.path.exists(gt_path)),
                         per_img_iou if per_img_iou is not None else "",
                         per_img_miou if per_img_miou is not None else "",
                         f"{infer_time:.6f}",
                         f"{total_time:.6f}"])

        print(f"[{base_name}] saved mask & overlay. has_gt={bool(gt_path)} infer_time={infer_time:.4f}s total_time={total_time:.4f}s")

    # Final FPS & IoU summary
    if infer_count > 0:
        infer_fps = infer_count / total_infer_time if total_infer_time > 0 else float('inf')
        total_fps = infer_count / total_total_time if total_total_time > 0 else float('inf')
    else:
        infer_fps = 0.0
        total_fps = 0.0

    # Compute final IoU across dataset
    ious_final, miou_final = compute_iou_from_accum(accum_intersections, accum_unions)

    summary_txt = os.path.join(output_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Num images processed: {infer_count}\n")
        f.write(f"Inference FPS (model forward only): {infer_fps:.2f}\n")
        f.write(f"Total FPS (including pre/post): {total_fps:.2f}\n")
        f.write("\nPer-class IoU:\n")
        for c in range(num_classes):
            iou_val = ious_final[c]
            if np.isnan(iou_val):
                f.write(f"  Class {c}: n/a (no samples)\n")
            else:
                f.write(f"  Class {c}: {iou_val:.6f}\n")
        f.write(f"\nMean IoU (valid classes): {miou_final:.6f}\n")
    print("---- Final results ----")
    print(f"Images processed: {infer_count}")
    print(f"Inference FPS: {infer_fps:.2f}")
    print(f"Total FPS: {total_fps:.2f}")
    for c in range(num_classes):
        print(f"Class {c} IoU: {('n/a' if np.isnan(ious_final[c]) else f'{ious_final[c]:.6f}')}")
    print(f"mIoU: {miou_final:.6f}")

    # Save CSV per-image summary
    csv_path = os.path.join(output_dir, "per_image_summary.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(csv_header)
        for row in csv_rows:
            writer.writerow(row)

    print(f"Saved summary: {summary_txt}")
    print(f"Saved per-image CSV: {csv_path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX DDRNet Inference with Overlay, IoU, and FPS")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    parser.add_argument("--gt_root", type=str, default="/home/trainer/DDRNet.pytorch/data/cityscapes/gtFine",
                        help="Root folder containing gtFine (or point directly to split folder)")
    parser.add_argument("--split", type=str, default="val", choices=["train","val","test"],
                        help="Which split subfolder under gt_root to use when searching for masks")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes (your model's classes)")
    parser.add_argument("--input_size", type=str, default="512,1024", help="Input size H,W used if model shape is ambiguous")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
    args = parser.parse_args()

    # parse input_size
    try:
        h_in, w_in = [int(x) for x in args.input_size.split(",")]
    except Exception:
        h_in, w_in = (512, 1024)

    infer_onnx(model_path=args.model,
               image_path=args.image,
               output_dir=args.output,
               gt_root=args.gt_root,
               split=args.split,
               num_classes=args.num_classes,
               input_size=(h_in, w_in),
               alpha=args.alpha)
