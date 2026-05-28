import argparse
import os
import pprint
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import _init_paths
import models
from config import config
from config import update_config
from datasets.base_dataset import BaseDataset
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


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
    parser = argparse.ArgumentParser(description='Sequential video inference for DDRNet.')
    parser.add_argument('--cfg', default='experiments/cityscapes/ddrnet23_slim.yaml')
    parser.add_argument('--video', required=True, help='Path to input video file.')
    parser.add_argument('--output-dir', default='', help='Directory to save output video overlays.')
    parser.add_argument('--alpha', default=0.4, type=float, help='Overlay alpha for segmentation mask.')
    parser.add_argument('--display-fps', action='store_true', help='Render FPS on output frames.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def maybe_reset_hidden(model):
    if hasattr(model, 'reset_hidden_state'):
        model.reset_hidden_state()
    elif hasattr(model, 'module') and hasattr(model.module, 'reset_hidden_state'):
        model.module.reset_hidden_state()


def load_model(cfg, final_output_dir, logger):
    module = eval('models.' + cfg.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval('models.' + cfg.MODEL.NAME + '.get_seg_model')(cfg)
    if cfg.TEST.MODEL_FILE:
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
    logger.info(f'=> loading model from {model_state_file}')

    state = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    model_dict = model.state_dict()
    state = {k[6:]: v for k, v in state.items() if k[6:] in model_dict}
    model_dict.update(state)
    model.load_state_dict(model_dict)

    model = nn.DataParallel(model).cuda()
    model.eval()
    maybe_reset_hidden(model)
    return model


def build_output_path(output_dir, video_path):
    output_dir = Path(output_dir) if output_dir else Path.cwd() / 'video_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{video_path.stem}_overlay.mp4'


def annotate_fps(frame, fps):
    text = f'FPS: {fps:.1f}'
    cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def process_video(video_path, model, dataset_helper, output_path, alpha, display_fps, logger):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f'Failed to open {video_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * 2, height),
    )

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = dataset_helper.input_transform(frame).transpose((2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float().cuda()

        with torch.no_grad():
            pred = model(image)
            if isinstance(pred, tuple) and len(pred) == 2:
                pred = pred[0]
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask_color = np.zeros((height, width, 3), dtype=np.uint8)
        mask_color[mask == 0] = (0, 255, 0)
        mask_color[mask == 1] = (255, 165, 0)
        mask_color[mask == 2] = (0, 0, 255)
        mask_color[mask == 3] = (0, 0, 0)
        mask_color[mask == 4] = (255, 0, 0)

        overlay = cv2.addWeighted(rgb.astype(np.uint8), 1.0 - alpha, mask_color, alpha, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        combined = np.concatenate([frame, overlay_bgr], axis=1)

        if display_fps:
            fps_value = frame_idx / max(1.0, time.time() - start_time)
            combined = annotate_fps(combined, fps_value)

        writer.write(combined)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info(f'Saved video inference output to {output_path}')


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'inference_video')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = load_model(config, final_output_dir, logger)
    dataset_helper = InferenceDatasetHelper(config)
    output_path = build_output_path(args.output_dir, Path(args.video))

    process_video(Path(args.video), model, dataset_helper, output_path, args.alpha, args.display_fps, logger)


if __name__ == '__main__':
    main()
