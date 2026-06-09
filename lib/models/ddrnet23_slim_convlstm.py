import torch
import torch.nn as nn

import models
from .convlstm import ConvLSTM2D


class DDRNetConvLSTM(nn.Module):
    """
    DDRNet backbone + ConvLSTM temporal fusion with optional keyframe processing.

    Keyframe mechanism:
        - Only run the expensive DDRNet backbone on keyframes (t % KEYFRAME_INTERVAL == 0)
        - For non-keyframes, reuse the most recent keyframe's features
        - ConvLSTM still processes every frame to maintain temporal state
        - Segment head produces output for every frame

    This reduces computation while preserving temporal information flow.
    """

    def __init__(self, base_model, convlstm_hidden_dim=64, keyframe_interval=1):
        super().__init__()
        self.base_model = base_model
        self.convlstm = None
        self.convlstm_hidden_dim = convlstm_hidden_dim
        self.hidden_state = None

        # Keyframe configuration
        self.keyframe_interval = max(1, int(keyframe_interval))

        # Delay construction until we see the first feature map
        self._feature_dim = None

        # Cache for keyframe features
        self._cached_features = None

        # Log keyframe configuration at startup
        print(f"[DDRNetConvLSTM] KEYFRAME_INTERVAL = {self.keyframe_interval}")

    def reset_hidden_state(self):
        """Reset both ConvLSTM hidden state and cached features."""
        self.hidden_state = None
        self._cached_features = None

    def _build_convlstm(self, feature_map):
        self._feature_dim = feature_map.size(1)
        self.convlstm = ConvLSTM2D(self._feature_dim, self.convlstm_hidden_dim, kernel_size=3)

    def _extract_features(self, x):
        """Run DDRNet backbone to extract spatial features from a single frame."""
        output = self.base_model(x)
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        return output, None

    def _forward_head(self, feature_map):
        """Apply segmentation head to ConvLSTM hidden state."""
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
        """
        Forward pass with keyframe-based feature extraction.

        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]
            hidden_state: Optional tuple (h, c) for ConvLSTM state

        Returns:
            outputs: Segmentation logits [B, T, num_outputs, C, H, W] or [B, num_outputs, C, H, W]
            hidden_state: Updated ConvLSTM state (h, c)
        """
        if hidden_state is None:
            hidden_state = self.hidden_state

        single_input = x.dim() == 4
        if single_input:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        B, T, C, H, W = x.shape
        outputs = []
        h, c = hidden_state if hidden_state is not None else (None, None)

        # Process each frame in the sequence
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]

            # KEYFRAME LOGIC:
            # Only run expensive DDRNet backbone on keyframes
            is_keyframe = (t % self.keyframe_interval == 0)

            if is_keyframe:
                # Extract fresh features from backbone
                features, _ = self._extract_features(frame)
                # Cache for subsequent non-keyframe frames
                self._cached_features = features
            else:
                # Reuse most recent keyframe features
                # This avoids the expensive backbone forward pass
                features = self._cached_features
                # Note: cached_features is guaranteed to exist because t=0 is always a keyframe

            # Initialize ConvLSTM on first frame (or when features first available)
            if self.convlstm is None:
                self._build_convlstm(features)

            # Initialize hidden state on first frame
            if h is None or c is None:
                h, c = self.convlstm.init_hidden(features)

            # ConvLSTM processes every frame to maintain temporal state
            # For non-keyframes, the same features are fed but hidden state evolves
            h, c = self.convlstm(features, (h, c))

            # Apply segmentation head to ConvLSTM hidden state
            out = self._forward_head(h)
            outputs.append(out)

        # Store hidden state for next sequence (detached to avoid gradient accumulation across sequences)
        self.hidden_state = (h.detach(), c.detach())
        hidden_state = (h, c)

        # Stack outputs: [B, T, num_outputs, C, H, W]
        outputs = torch.stack(outputs, dim=1)

        if single_input:
            outputs = outputs[:, 0]

        return outputs, hidden_state


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

    # Read KEYFRAME_INTERVAL from config (default=1 for backward compatibility)
    keyframe_interval = getattr(config.MODEL, 'KEYFRAME_INTERVAL', 1)

    return DDRNetConvLSTM(
        base_model,
        convlstm_hidden_dim=getattr(config.MODEL, 'CONVLSTM_HIDDEN_DIM', 64),
        keyframe_interval=keyframe_interval
    )
