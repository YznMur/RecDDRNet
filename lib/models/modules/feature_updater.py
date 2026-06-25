import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureUpdater(nn.Module):
    """
    Lightweight feature updater that estimates a feature correction delta_F
    from the difference between consecutive RGB frames, without running the
    full DDRNet backbone.

    Input:
        prev_frame:    [B, 3, H, W]  (previous RGB frame, normalized)
        current_frame: [B, 3, H, W]  (current RGB frame, normalized)
        prev_features: [B, C, h, w]  (previous feature map at 1/8 res)

    Output:
        delta_F: [B, C, h, w]  (feature correction to add to prev_features)

    Architecture:
        concat(prev_frame, current_frame) -> [B, 6, H, W]
        4x Conv3x3-BN-ReLU (stride 2 progressively downsamples to 1/8 res)
        Conv 1x1 projection to C channels
    """

    def __init__(self, feature_channels=128, hidden_channels=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, prev_frame, current_frame, prev_features):
        x = torch.cat([prev_frame, current_frame], dim=1)
        x = self.encoder(x)

        h, w = prev_features.shape[2], prev_features.shape[3]
        if x.shape[2] != h or x.shape[3] != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        delta_f = self.project(x)
        return delta_f
