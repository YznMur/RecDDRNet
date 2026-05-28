import argparse
import os
import pprint
import timeit
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import _init_paths
import models
from config import config
from config import update_config
from datasets.base_dataset import BaseDataset
from utils.modelsummary import get_model_summary
from utils.utils import Map16, create_logger


# -------------------------------
# LEGEND CONFIG
# -------------------------------
CLASS_NAMES = [
    "Field",
    "Grass",
    "Windrow",
    "Unlabeled objects",
    "Obstacles",
]

CLASS_COLORS = np.array([
    [0, 255, 0],     # Field - green
    [255, 165, 0],   # Grass - orange
    [0, 0, 255],     # Windrow - blue
    [0, 0, 0],       # Unlabeled - black
    [255, 0, 0],     # Obstacles - red
])


def draw_legend(image, names, colors, x=10, y=10):
    """
    Draws a legend on the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    box_size = 15
    line_height = 22

    # Background for readability
    cv2.rectangle(
        image,
        (x - 5, y - 5),
        (x + 200, y + len(names) * line_height),
        (0, 0, 0),
        -1
    )

    for i, (name, color) in enumerate(zip(names, colors)):
        y_offset = y + i * line_height

        # Color box
        cv2.rectangle(
            image,
            (x, y_offset),
            (x + box_size, y_offset + box_size),
            color.tolist(),
            -1
        )

        # Label text
        cv2.putText(
            image,
            name,
            (x + box_size + 8, y_offset + box_size - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return image


class InferenceDatasetHelper(BaseDataset):
    def __init__(self, cfg):
        test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
        super().__init__(
            ignore_label=cfg.TRAIN.IGNORE_LABEL,
            base_size=cfg.TEST.BASE_SIZE,
            crop_size=test_size,
            downsample_rate=1,
        )
        self.num_classes = cfg.DATASET.NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run segmentation inference on videos in a folder."
    )
    parser.add_argument("--cfg", default="experiments/cityscapes/ddrnet23_slim.yaml")
    parser.add_argument("--input-folder", required=True, type=str)
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def load_model(cfg, final_output_dir, logger):
    module = eval("models." + cfg.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval("models." + cfg.MODEL.NAME + ".get_seg_model")(cfg)

    dump_input = torch.rand((1, 3, cfg.TRAIN.IMAGE_SIZE[1], cfg.TRAIN.IMAGE_SIZE[0]))
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    model_state_file = (
        cfg.TEST.MODEL_FILE
        if cfg.TEST.MODEL_FILE
        else os.path.join(final_output_dir, "checkpoint.pth.tar")
    )

    logger.info(f"=> loading model from {model_state_file}")

    pretrained_dict = torch.load(model_state_file)
    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]

    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def build_output_dir(args, final_output_dir, video_name):
    if args.output_dir:
        output_dir = Path(args.output_dir) / video_name
    else:
        output_dir = Path(final_output_dir) / "video_overlays" / video_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_video_files(folder):
    exts = [".mp4", ".avi", ".mov", ".mkv"]
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]


def format_timestamp(frame_idx, fps):
    total_ms = int(round((frame_idx / fps) * 1000.0)) if fps > 0 else 0
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}-{m:02d}-{s:02d}-{ms:03d}"


def save_overlay_frame(output_dir, video_name, frame_idx, fps, overlay):
    ts = format_timestamp(frame_idx, fps)
    filename = f"{video_name}_frame{frame_idx:06d}_{ts}.jpg"
    save_path = output_dir / filename
    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def process_video(video_path, model, dataset_helper, args, logger, final_output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.stem

    frames_output_dir = build_output_dir(args, final_output_dir, video_name)
    output_video_path = frames_output_dir.parent / f"{video_name}_preview.mp4"

    map16 = Map16(None)

    ret, frame = cap.read()
    if not ret:
        return

    h, w = frame.shape[:2]

    out = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        160,
        (w * 2, h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    start = timeit.default_timer()

    with torch.no_grad():
        for frame_idx in tqdm(range(frame_count), desc=video_name):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            original = frame_bgr.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            transformed = dataset_helper.input_transform(frame_bgr).transpose((2, 0, 1))
            image = torch.from_numpy(np.expand_dims(transformed, axis=0)).float()

            pred = dataset_helper.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST,
            )

            if pred.size()[-2:] != frame_rgb.shape[:2]:
                pred = F.interpolate(
                    pred,
                    size=frame_rgb.shape[:2],
                    mode="bilinear",
                    align_corners=config.MODEL.ALIGN_CORNERS,
                )

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_color = map16.colors[pred]

            overlay = cv2.addWeighted(
                frame_rgb.astype(np.uint8),
                1.0 - args.alpha,
                pred_color.astype(np.uint8),
                args.alpha,
                0,
            )

            # -------------------------------
            # DRAW LEGEND HERE
            # -------------------------------
            overlay = draw_legend(overlay, CLASS_NAMES, CLASS_COLORS)

            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            preview = np.concatenate([original, overlay_bgr], axis=1)
            out.write(preview)
            # save_overlay_frame(frames_output_dir, video_name, frame_idx, fps, overlay)

    cap.release()
    out.release()

    logger.info(f"Saved preview → {output_video_path}")
    logger.info(f"Time: {(timeit.default_timer() - start)/60:.2f} min")


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, "video_demo")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = load_model(config, final_output_dir, logger)
    dataset_helper = InferenceDatasetHelper(config)

    video_files = get_video_files(args.input_folder)

    if not video_files:
        logger.warning("No videos found.")
        return

    logger.info(f"Found {len(video_files)} videos")

    for video_path in video_files:
        logger.info(f"Processing: {video_path}")
        process_video(video_path, model, dataset_helper, args, logger, final_output_dir)

    logger.info("All videos processed.")


if __name__ == "__main__":
    main()