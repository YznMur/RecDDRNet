#!/usr/bin/env python3
import os
import cv2
import numpy as np
import onnxruntime
import argparse
import time

# -----------------------------
# Utilities
# -----------------------------
def preprocess_image(img, target_size):
    H, W = target_size
    img_resized = cv2.resize(img, (W, H))
    img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    img_chw = img_norm.transpose(2, 0, 1)
    return np.expand_dims(img_chw, axis=0).astype(np.float32)

def get_colormap(n_classes):
    base = [
        [34, 139, 34],    # forest green
        [0, 100, 200],      # dark green
        [500, 5, 128],    # teal
        [25, 25, 112],    # midnight blue
        [0, 0, 139],      # dark blue
        [0, 51, 102],     # navy blue
        [46, 139, 87],    # sea green
        [0, 70, 140],     # deep steel blue
    ]
    if n_classes > len(base):
        base = (base * ((n_classes // len(base)) + 1))[:n_classes]
    return np.array(base[:n_classes], dtype=np.uint8)

def mask_to_color(mask, colormap):
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        color_mask[mask == i] = color
    return color_mask


def overlay_mask_on_image(image, mask, alpha=0.6):
    return cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0)


# -----------------------------
# Legend drawing
# -----------------------------
def draw_legend(
    image,
    colormap,
    class_names=None,
    origin=(20, 20),
    box_size=20,
    spacing=8
):
    """
    Draws a legend on the image (top-left).
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(colormap))]

    x0, y0 = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Compute legend box size
    legend_height = len(class_names) * (box_size + spacing) + spacing
    legend_width = 0
    for name in class_names:
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        legend_width = max(legend_width, box_size + spacing + tw)
    legend_width += spacing * 3

    # Background (black box)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x0 - 10, y0 - 10),
        (x0 + legend_width, y0 + legend_height),
        (0, 0, 0),
        -1
    )
    image[:] = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Draw entries
    y = y0
    for idx, (color, name) in enumerate(zip(colormap, class_names)):
        # Color box
        cv2.rectangle(
            image,
            (x0, y),
            (x0 + box_size, y + box_size),
            tuple(int(c) for c in color),
            -1
        )
        # Text
        cv2.putText(
            image,
            name,
            (x0 + box_size + spacing, y + box_size - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        y += box_size + spacing

    return image


# -----------------------------
# Inference
# -----------------------------
def infer_onnx_single_image(
    model_path,
    image_path,
    output_dir="output",
    num_classes=4,
    input_size=(512, 1024),
    alpha=0.6,
    providers=["CPUExecutionProvider"]
):
    os.makedirs(output_dir, exist_ok=True)

    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_meta = session.get_outputs()[0]

    in_shape = session.get_inputs()[0].shape
    H_in = in_shape[2] if in_shape[2] is not None else input_size[0]
    W_in = in_shape[3] if in_shape[3] is not None else input_size[1]

    try:
        if output_meta.shape[1] is not None:
            num_classes = int(output_meta.shape[1])
    except Exception:
        pass

    colormap = get_colormap(num_classes)
    class_names = [f"Class {i}" for i in range(num_classes)]

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    H_orig, W_orig = img.shape[:2]

    t0 = time.perf_counter()
    x = preprocess_image(img, (H_in, W_in))

    t_inf_start = time.perf_counter()
    pred = session.run(None, {input_name: x})[0]
    t_inf_end = time.perf_counter()

    if pred.ndim == 4:
        pred_mask_small = np.argmax(pred[0], axis=0).astype(np.uint8)
    else:
        pred_mask_small = np.argmax(pred, axis=0).astype(np.uint8)

    pred_mask = cv2.resize(
        pred_mask_small,
        (W_orig, H_orig),
        interpolation=cv2.INTER_NEAREST
    )

    color_mask = mask_to_color(pred_mask, colormap)
    overlay = overlay_mask_on_image(img, color_mask, alpha)

    # Draw legend ON overlay
    overlay = draw_legend(overlay, colormap, class_names)

    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)

    mask_path = os.path.join(output_dir, f"{name}_mask.png")
    overlay_path = os.path.join(output_dir, f"{name}_overlay.png")

    cv2.imwrite(mask_path, color_mask[:, :, ::-1])
    cv2.imwrite(overlay_path, overlay)

    print("---- Done ----")
    print(f"Inference time: {(t_inf_end - t_inf_start):.4f}s")
    print(f"Saved mask   : {mask_path}")
    print(f"Saved overlay: {overlay_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ONNX segmentation inference (single image + legend)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--input_size", default="512,1024")
    parser.add_argument("--alpha", type=float, default=0.6)

    args = parser.parse_args()
    h, w = [int(x) for x in args.input_size.split(",")]

    infer_onnx_single_image(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        num_classes=args.num_classes,
        input_size=(h, w),
        alpha=args.alpha
    )
