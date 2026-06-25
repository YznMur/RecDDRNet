import torch
import torch.nn as nn

import models
from .convlstm import ConvLSTM2D
from .modules.feature_updater import FeatureUpdater


class DDRNetConvLSTM(nn.Module):
    """
    DDRNet backbone + ConvLSTM temporal fusion with optional keyframe processing
    and optional lightweight feature updater.

    Keyframe mechanism:
        - Only run the expensive DDRNet backbone on keyframes (t % KEYFRAME_INTERVAL == 0)
        - For non-keyframes with updater: FeatureUpdater estimates delta_F from frame pairs
        - For non-keyframes without updater: reuse cached keyframe features
        - ConvLSTM still processes every frame to maintain temporal state
        - Segment head produces output for every frame
    """

    def __init__(self, base_model, convlstm_hidden_dim=64, keyframe_interval=1,
                 use_feature_updater=False, updater_channels=64):
        super().__init__()
        self.base_model = base_model
        self.convlstm = None
        self.convlstm_hidden_dim = convlstm_hidden_dim
        self.hidden_state = None

        # Keyframe configuration
        self.keyframe_interval = max(1, int(keyframe_interval))

        # Feature updater configuration
        self.use_feature_updater = use_feature_updater
        self._updater_channels = updater_channels

        # Delay construction until we see the first feature map
        self._feature_dim = None

        # Cache for keyframe features
        self._cached_features = None

        # Updater state
        self._prev_frame = None
        self._prev_features = None
        self.feature_updater = None

        print(f"[DDRNetConvLSTM] KEYFRAME_INTERVAL = {self.keyframe_interval}")
        print(f"[DDRNetConvLSTM] USE_FEATURE_UPDATER = {self.use_feature_updater}")

    def reset_hidden_state(self):
        """Reset ConvLSTM hidden state, cached features, and updater state."""
        self.hidden_state = None
        self._cached_features = None
        self._prev_frame = None
        self._prev_features = None

    def _build_convlstm(self, feature_map):
        self._feature_dim = feature_map.size(1)
        self.convlstm = ConvLSTM2D(self._feature_dim, self.convlstm_hidden_dim, kernel_size=3)
        if self.use_feature_updater and self.feature_updater is None:
            self.feature_updater = FeatureUpdater(
                feature_channels=self._feature_dim,
                hidden_channels=self._updater_channels,
            ).to(feature_map.device)

    def _extract_features(self, x):
        """Run DDRNet backbone to extract spatial features from a single frame."""
        output = self.base_model(x)
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            return output[0], output[1]
        return output, None

    def _forward_head(self, feature_map):
        """Apply segmentation head to ConvLSTM hidden state."""
        if hasattr(self.base_model, 'final_layer'):
            return self.base_model.final_layer(feature_map)
        if hasattr(self.base_model, 'head'):
            return self.base_model.head(feature_map)
        if hasattr(self.base_model, 'cls_seg'):
            return self.base_model.cls_seg(feature_map)
        if hasattr(self.base_model, 'seg_head'):
            return self.base_model.seg_head(feature_map)
        raise AttributeError(
            'Base DDRNet model does not expose a known head module (final_layer/head/cls_seg/seg_head).'
        )

    def forward(self, x, hidden_state=None):
        """
        Forward pass with keyframe-based feature extraction and optional feature updater.

        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]
            hidden_state: Optional tuple (h, c) for ConvLSTM state

        Returns:
            outputs: (main_out, extra_out) tuple of [B, T, C, H, W]
            hidden_state: Updated ConvLSTM state (h, c)
        """
        if hidden_state is None:
            hidden_state = self.hidden_state

        single_input = x.dim() == 4
        if single_input:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        B, T, C, H, W = x.shape
        main_outputs = []
        extra_outputs = []
        h, c = hidden_state if hidden_state is not None else (None, None)

        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            is_keyframe = (t % self.keyframe_interval == 0)

            if is_keyframe or self._cached_features is None:
                features, extra_out = self._extract_features(frame)
                self._cached_features = features
                self._cached_extra = extra_out
            else:
                if self.use_feature_updater and self.feature_updater is not None and self._prev_frame is not None:
                    delta_f = self.feature_updater(self._prev_frame, frame, self._prev_features)
                    features = self._prev_features + delta_f
                else:
                    features = self._cached_features
                extra_out = self._cached_extra

            if self.convlstm is None:
                self._build_convlstm(features)

            if h is None or c is None:
                h, c = self.convlstm.init_hidden(features)

            h, c = self.convlstm(features, (h, c))
            main_out = self._forward_head(h)
            main_outputs.append(main_out)
            extra_outputs.append(extra_out)

            self._prev_frame = frame.detach()
            self._prev_features = features.detach()
            self._cached_features = features.detach() if not is_keyframe else self._cached_features
            self._cached_extra = extra_out.detach() if not is_keyframe else self._cached_extra

        self.hidden_state = (h.detach(), c.detach())
        hidden_state = (h, c)

        main_outputs = torch.stack(main_outputs, dim=1)
        extra_outputs = torch.stack(extra_outputs, dim=1)

        if single_input:
            main_outputs = main_outputs[:, 0]
            extra_outputs = extra_outputs[:, 0]

        return [extra_outputs, main_outputs], hidden_state


def get_seg_model(config):
    base_name = config.MODEL.NAME
    if base_name.endswith('_convlstm'):
        base_name = base_name[:-len('_convlstm')]

    # Normalize model naming conventions between configs and module names
    if not hasattr(models, base_name):
        if base_name.startswith('ddrnet23_'):
            base_name = base_name.replace('ddrnet23_', 'ddrnet_23_', 1)
        elif base_name == 'ddrnet23':
            base_name = 'ddrnet_23'
        elif base_name.startswith('ddrnet39'):
            base_name = base_name.replace('ddrnet39', 'ddrnet_39', 1)

    base_model = eval(f'models.{base_name}.get_seg_model')(config)

    # Read config
    keyframe_interval = getattr(config.MODEL, 'KEYFRAME_INTERVAL', 1)
    use_feature_updater = getattr(config.MODEL, 'USE_FEATURE_UPDATER', False)
    updater_channels = getattr(config.MODEL, 'UPDATER_CHANNELS', 64)

    return DDRNetConvLSTM(
        base_model,
        convlstm_hidden_dim=getattr(config.MODEL, 'CONVLSTM_HIDDEN_DIM', 64),
        keyframe_interval=keyframe_interval,
        use_feature_updater=use_feature_updater,
        updater_channels=updater_channels,
    )
