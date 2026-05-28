import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
from config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Export DDRNet to ONNX')
    parser.add_argument('--cfg', default="experiments/rsm/ddrnet23.yaml", type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def load_ddrnet_model():
    module = eval('models.' + config.MODEL.NAME)

    # Fix batchnorm for torch1.x
    if torch.__version__.startswith("1"):
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    # Load weights
    model_file = config.TEST.MODEL_FILE or os.path.join(config.OUTPUT_DIR, "rsm/ddrnet23/checkpoint.pth.tar")
    print("Loading:", model_file)

    pretrained = torch.load(model_file, map_location="cpu")
    if 'state_dict' in pretrained:
        pretrained = pretrained["state_dict"]

    # remove "model." prefix
    pretrained = {k[6:]: v for k, v in pretrained.items() if k[6:] in model.state_dict()}
    model.load_state_dict({**model.state_dict(), **pretrained})
    return model


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, out = self.model(x)  # fused output
        # Upsample to the input size
        out_upsampled = nn.functional.interpolate(
            out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False
        )
        return out_upsampled


def main():
    args = parse_args()

    cudnn.benchmark = True

    # Load original model
    model = load_ddrnet_model().eval()

    # Wrap to export only fused output (index=1)
    net = ONNXWrapper(model).eval()

    # Fixed resolution from config
    # W, H = config.TEST.IMAGE_SIZE  # e.g. (2048, 1024)
    # dummy = torch.randn(1, 3, H, W)
    # dummy = torch.randn(1, 3, 512, 1024)
    dummy = torch.randn(1, 3, 480, 640)

    output_path = f"rsm_ddrnet23_output/{config.MODEL.NAME}_fp32_bs1.onnx"
    os.makedirs("rsm_ddrnet23_output", exist_ok=True)

    print("Export to:", output_path)

    torch.onnx.export(
        net,
        dummy,
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["segmentation"],
        dynamic_axes=None,   # fixed size for TensorRT
        verbose=False
    )

    print("ONNX export done.")


if __name__ == "__main__":
    main()
