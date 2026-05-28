import argparse
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import onnx

import _init_paths
import models
from config import config, update_config
from utils.utils import create_logger


class onnx_net(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def forward(self, x):
        x1, x2 = self.backbone(x)
        y = F.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        # y = F.softmax(y, dim=1)  # probabilities, not argmax
        return y


def parse_args():
    parser = argparse.ArgumentParser(description='Export DDRNet ONNX')
    parser.add_argument('--cfg', default="experiments/cityscapes/ddrnet23_slim.yaml", type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'export')

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Build model
    module = eval(f"models.{config.MODEL.NAME}")
    model = eval(f"models.{config.MODEL.NAME}.get_seg_model")(config)

    # Load weights
    model_file = config.TEST.MODEL_FILE
    logger.info(f"=> loading model from {model_file}")
    pretrained_dict = torch.load(model_file, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    net = onnx_net(model).eval()

    # Dummy input (1 × 3 × 1024 × 2048)
    dummy_input = torch.randn((1, 3, 1024, 2048))

    output_dir = "output_onnx_rsm_imagenet_10cls"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ddrnet23_slim_cityscapes.onnx")

    torch.onnx.export(
        net,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['inputx'],
        output_names=['outputy'],
        dynamic_axes={
            'inputx': {2: 'height', 3: 'width'},
            'outputy': {2: 'height', 3: 'width'},
        },
        verbose=True
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"✅ Exported ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
