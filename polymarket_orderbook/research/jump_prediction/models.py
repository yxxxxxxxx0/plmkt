"""Phase 5: sequence models over the raw LOB, sections 8-10 of the brief.

Three models share ONE encoder so the comparison isolates the temporal part:

  CNNOnly          encoder -> temporal convolution -> pooled -> head
  CNNLSTM          encoder -> 1-layer LSTM -> last state -> head
  CNNTransformer   encoder -> 2-layer transformer encoder -> pooled -> head

The encoder reads a frame as a (channels x levels) image and convolves along
the LEVEL axis, so the arrangement of the book is preserved rather than
flattened. Channels per side are price distance from mid, log dollars, and the
existence mask -- the mask is a first-class input because a book with three
levels is genuinely different from a book with ten, and without it the model
would read a padded level as a real quote.

Elapsed time. The brief asks for delta_t because sampling may be irregular. On
the 200ms grid the spacing between frames is constant by construction, so a
literal delta_t channel would be a constant and carry nothing. What DOES vary,
and what actually encodes irregularity here, is how stale each frame's book
is: `book_age` says how long ago the exchange last updated that book. That is
the channel used, and the "no delta_t" ablation removes it.

Sizes are deliberately small -- the aim is to test a hypothesis, not to
maximise capacity. Each model is roughly 60-160k parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# per-frame LOB channels: bid dist, bid usd, bid mask, ask dist, ask usd, ask mask
N_CH = 6
# per-frame scalars: mid offset from the prediction instant, spread, book age
N_SCALAR = 3


class LOBEncoder(nn.Module):
    """One frame -> one embedding. Convolves across levels, not across time.

    Kernel (1, 2) with stride (1, 2) first mixes the (distance, size) pair at
    each level in the style of DeepLOB, then successive convolutions widen
    across levels. Nothing here touches the time axis, so every temporal model
    below sees identical per-frame features.
    """

    def __init__(self, k_levels=10, d=64, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        c_in = N_CH if use_mask else 4
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (1, 3), padding=(0, 1)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, (1, 3), padding=(0, 1)), nn.LeakyReLU(0.01),
        )
        self.out_dim = 64 * (k_levels // 2)
        self.proj = nn.Linear(self.out_dim + N_SCALAR, d)

    def forward(self, x, s):
        # x: (B, L, C, K)   s: (B, L, N_SCALAR)
        B, L, C, K = x.shape
        if not self.use_mask:
            x = x[:, :, [0, 1, 3, 4], :]
            C = 4
        h = self.net(x.reshape(B * L, C, 1, K))
        h = h.reshape(B * L, -1)
        h = torch.cat([h, s.reshape(B * L, -1)], dim=-1)
        return self.proj(h).reshape(B, L, -1)


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2),
                                 nn.LeakyReLU(0.01), nn.Dropout(0.1),
                                 nn.Linear(d // 2, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class CNNOnly(nn.Module):
    """No recurrence and no attention: a temporal convolution then a pool.

    This is the control for the whole study. If it matches the LSTM and the
    transformer, then ordering and long-range comparison across the window are
    not carrying anything, and the extra machinery is unjustified.
    """

    def __init__(self, k_levels=10, d=64, use_mask=True):
        super().__init__()
        self.enc = LOBEncoder(k_levels, d, use_mask)
        self.tcn = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2), nn.LeakyReLU(0.01),
            nn.Conv1d(d, d, 5, padding=2, dilation=1), nn.LeakyReLU(0.01))
        self.head = Head(d * 2)

    def forward(self, x, s):
        h = self.enc(x, s).transpose(1, 2)
        h = self.tcn(h)
        h = torch.cat([h.mean(-1), h[:, :, -1]], dim=-1)
        return self.head(h)


class CNNLSTM(nn.Module):
    def __init__(self, k_levels=10, d=64, use_mask=True):
        super().__init__()
        self.enc = LOBEncoder(k_levels, d, use_mask)
        self.rnn = nn.LSTM(d, d, num_layers=1, batch_first=True)
        self.head = Head(d)

    def forward(self, x, s):
        h, _ = self.rnn(self.enc(x, s))
        return self.head(h[:, -1])


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of elapsed seconds before the prediction instant.

    Uses the actual time offset rather than the token index, so the model is
    told when each frame happened. On a uniform grid these coincide; keeping
    it explicit means the same code stays correct if the event-level track,
    where spacing really is irregular, is fed through it.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d
        inv = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        self.register_buffer("inv", inv)

    def forward(self, t):                       # t: (B, L) seconds, <= 0
        a = t.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)[..., :self.d]


class CNNTransformer(nn.Module):
    def __init__(self, k_levels=10, d=64, use_mask=True, nhead=4, layers=2):
        super().__init__()
        self.enc = LOBEncoder(k_levels, d, use_mask)
        self.time = TimeEncoding(d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                           dim_feedforward=d * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = Head(d * 2)

    def forward(self, x, s, t=None):
        h = self.enc(x, s)
        if t is not None:
            h = h + self.time(t)
        h = self.tr(h)
        return self.head(torch.cat([h.mean(1), h[:, -1]], dim=-1))


def build(name, k_levels=10, d=64, use_mask=True):
    m = {"cnn": CNNOnly, "cnn_lstm": CNNLSTM,
         "cnn_transformer": CNNTransformer}[name](k_levels, d, use_mask)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
