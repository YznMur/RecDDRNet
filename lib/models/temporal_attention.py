import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    """
    Lightweight temporal attention block (TAB) that fuses current-frame features with
    a compact memory of previous-frame features.

    - Operates at 1/8 feature resolution: input `x` shape [B, C, H, W]
    - Keeps channel dimension small (<=128) for real-time performance
    - Uses per-spatial-location temporal attention (no spatial-token quadratic cost)
    - Memory is a single-frame feature tensor (previous aggregated features)

    Forward inputs:
      x: current features [B, C, H, W]
      memory: previous features [B, C, H, W] or None

    Returns:
      out: refined features [B, C, H, W]
      new_memory: updated memory [B, C, H, W]

    Implementation notes:
      - Projects to multiple heads with 1x1 convs for Q/K/V
      - Computes similarity per-head per-spatial-location between current Q and prev K
      - Uses a sigmoid gating (lightweight) instead of full softmax over long sequences
      - Fuses gated V into the current feature with a residual connection
      - Memory update is a simple EMA for streaming efficiency
    """

    def __init__(self, channels, heads=4, head_dim=None, memory_ema=0.9):
        super().__init__()
        self.channels = channels
        self.heads = heads
        if head_dim is None:
            assert channels % heads == 0, "channels must be divisible by heads"
            head_dim = channels // heads
        self.head_dim = head_dim
        assert heads * head_dim <= 128, "keep heads*head_dim <= 128 for efficiency"

        self.to_q = nn.Conv2d(channels, heads * head_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, heads * head_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, heads * head_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(heads * head_dim, channels, kernel_size=1, bias=False)

        self.norm = nn.BatchNorm2d(channels)
        self.memory_ema = float(memory_ema)

    def forward(self, x, memory=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        q = self.to_q(x).view(B, self.heads, self.head_dim, H, W)

        if memory is None:
            # initialize memory as current features (lightweight)
            k = self.to_k(x).view(B, self.heads, self.head_dim, H, W)
            v = self.to_v(x).view(B, self.heads, self.head_dim, H, W)
            # no gating needed, return identity with memory
            out = self.out_proj(v.view(B, self.heads * self.head_dim, H, W))
            out = self.norm(out + x)
            new_memory = x.detach()
            return out, new_memory

        # Project memory
        k = self.to_k(memory).view(B, self.heads, self.head_dim, H, W)
        v = self.to_v(memory).view(B, self.heads, self.head_dim, H, W)

        # similarity per-head per-spatial-location: [B, heads, H, W]
        # dot-product along head_dim
        sim = (q * k).sum(dim=2) / math.sqrt(self.head_dim)

        # lightweight gating across time (current vs previous influence)
        gate = torch.sigmoid(sim).unsqueeze(2)  # [B, heads, 1, H, W]

        # gated value: [B, heads, head_dim, H, W]
        gated_v = gate * v

        # merge heads back: [B, heads*head_dim, H, W]
        merged = gated_v.reshape(B, self.heads * self.head_dim, H, W)
        fused = self.out_proj(merged)

        # residual refinement and normalization
        out = self.norm(fused + x)

        # update memory with EMA on feature tensor (not the projected v) to keep memory compact
        new_memory = (self.memory_ema * memory) + ((1.0 - self.memory_ema) * x.detach())

        return out, new_memory


if __name__ == '__main__':
    # quick smoke test
    m = TemporalAttentionBlock(channels=64, heads=4)
    x = torch.randn(2, 64, 32, 32)
    out, mem = m(x, None)
    print(out.shape, mem.shape)