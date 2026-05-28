# ------------------------------------------------------------------------------
# Demo script for segmentation inference on a folder of images
# Outputs: side-by-side (original | mask | overlay)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
from config import config
from config import update_config
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Demo')

    parser.add_argument('--cfg',
                        help='experiment config file',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)

    parser.add_argument('--input',
                        help='input image folder',
                        required=True,
                        type=str)

    parser.add_argument('--output',
                        help='output folder',
                        default='demo_outputs',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


# ------------------------------------------------------------------------------
# Color map (adjust to your dataset if needed)
# ------------------------------------------------------------------------------
def get_color_map():
    return np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
    ])


# ------------------------------------------------------------------------------
# Put label text on image
# ------------------------------------------------------------------------------
def put_text(img, text):
    return cv2.putText(img.copy(), text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2, cv2.LINE_AA)


# ------------------------------------------------------------------------------
# Demo function
# ------------------------------------------------------------------------------
def run_demo(model, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    color_map = get_color_map()
    image_paths = list(Path(image_folder).glob("*.*"))

    print(f"Found {len(image_paths)} images")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping {img_path}")
            continue

        orig = img.copy()

        # Resize to model input
        img_resized = cv2.resize(img, tuple(config.TEST.IMAGE_SIZE))
        img_resized = img_resized.astype(np.float32) / 255.0

        # HWC → CHW
        img_resized = img_resized.transpose(2, 0, 1)

        # Normalize (IMPORTANT if used in training)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            img_resized[i] = (img_resized[i] - mean[i]) / std[i]

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).cuda()

        # Inference
        with torch.no_grad():
            pred = model(img_tensor)

            if isinstance(pred, list):
                pred = pred[config.TEST.OUTPUT_INDEX]

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        # Resize prediction back
        pred = cv2.resize(
            pred.astype(np.uint8),
            (orig.shape[1], orig.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Color mask
        color_mask = color_map[pred].astype(np.uint8)

        # Overlay
        overlay = cv2.addWeighted(orig, 0.6, color_mask, 0.4, 0)

        # Add labels
        orig_l = put_text(orig, "Original")
        mask_l = put_text(color_mask, "Mask")
        overlay_l = put_text(overlay, "Overlay")

        # Optional spacing
        gap = np.ones((orig.shape[0], 10, 3), dtype=np.uint8) * 255

        # Combine
        combined = np.concatenate(
            [orig_l, gap, mask_l, gap, overlay_l],
            axis=1
        )

        # Save
        save_path = os.path.join(output_folder, img_path.stem + "_viz.png")
        cv2.imwrite(save_path, combined)

        print(f"Saved: {save_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'demo')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # CUDNN settings
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    # Model summary
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # Load weights
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')

    logger.info(f'=> loading model from {model_state_file}')

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()

    pretrained_dict = {
        k[6:]: v for k, v in pretrained_dict.items()
        if k[6:] in model_dict.keys()
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Multi-GPU
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Run demo
    run_demo(model, args.input, args.output)

    print("Done!")


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    main()







# python demo_preview.py \
#   --cfg experiments/cityscapes/ddrnet23_slim_640x480_5classes.yaml \
#   --input demo_images \
#   --output demo_outputs_ddrnet23_slim_640x480_5classes