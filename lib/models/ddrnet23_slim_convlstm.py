import torch
import torch.nn as nn

import models
from .convlstm import ConvLSTM2D


class DDRNetConvLSTM(nn.Module):
    def __init__(self, base_model, convlstm_hidden_dim=64):
        super().__init__()
        self.base_model = base_model
        self.convlstm = None
        self.convlstm_hidden_dim = convlstm_hidden_dim
        self.hidden_state = None

        # Delay construction until we see the first feature map
        self._feature_dim = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def _build_convlstm(self, feature_map):
        self._feature_dim = feature_map.size(1)
        self.convlstm = ConvLSTM2D(self._feature_dim, self.convlstm_hidden_dim, kernel_size=3)

    def _extract_features(self, x):
        output = self.base_model(x)
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        return output, None

    def _forward_head(self, feature_map):
        if hasattr(self.base_model, 'head'):
            return self.base_model.head(feature_map)
        if hasattr(self.base_model, 'cls_seg'):
            return self.base_model.cls_seg(feature_map)
        if hasattr(self.base_model, 'seg_head'):
            return self.base_model.seg_head(feature_map)
        raise AttributeError(
            'Base DDRNet model does not expose a known head module (head/cls_seg/seg_head) for ConvLSTM integration.'
        )

    def forward(self, x, hidden_state=None):
        """Accepts x:[B,C,H,W] or x:[B,T,C,H,W].

        Returns segmentation logits and the updated hidden state tuple.
        """
        if hidden_state is None:
            hidden_state = self.hidden_state

        single_input = x.dim() == 4
        if single_input:
            x = x.unsqueeze(1)

        outputs = []
        h, c = hidden_state if hidden_state is not None else (None, None)

        for t in range(x.size(1)):
            frame = x[:, t]
            features, _ = self._extract_features(frame)
            if self.convlstm is None:
                self._build_convlstm(features)
            if h is None or c is None:
                h, c = self.convlstm.init_hidden(features)
            h, c = self.convlstm(features, (h, c))
            out = self._forward_head(h)
            outputs.append(out)

        self.hidden_state = (h.detach(), c.detach())
        hidden_state = (h, c)

        outputs = torch.stack(outputs, dim=1)
        if single_input:
            outputs = outputs[:, 0]

        return outputs, hidden_state


def get_seg_model(config):
    base_name = config.MODEL.NAME
    if base_name.endswith('_convlstm'):
        base_name = base_name[:-len('_convlstm')]

    base_model = eval(f'models.{base_name}.get_seg_model')(config)
    return DDRNetConvLSTM(base_model, convlstm_hidden_dim=64)
