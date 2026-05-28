import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Lightweight ConvLSTM cell for feature-level temporal fusion."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, hx=None):
        # x: [B, C, H, W]
        # hx: tuple(h, c) each [B, hidden_dim, H, W]
        if hx is None:
            hx = self.init_hidden(x)
        h_prev, c_prev = hx

        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)

        ci, cf, co, cg = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        g = torch.tanh(cg)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, x, batch_size=None):
        if batch_size is None:
            batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        device = x.device
        dtype = x.dtype

        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        return h, c


class ConvLSTM2D(nn.Module):
    """Wrapper around ConvLSTMCell to support optional sequence input."""

    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, bias=True):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size=kernel_size, bias=bias)

    def forward(self, x, hx=None):
        return self.cell(x, hx)

    def init_hidden(self, x, batch_size=None):
        return self.cell.init_hidden(x, batch_size=batch_size)
