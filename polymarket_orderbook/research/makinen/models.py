"""Stage 7: the model zoo, in increasing order of what it is allowed to see.

Model 0  prevalence            constant = training base rate
Model 1  time-of-day only      logistic on tod_sin/tod_cos (+ dow)
Model 2  logistic              latest snapshot x_t, linear
Model 3  MLP snapshot          x_t only
Model 4  MLP history           flattened [L, F]
Model 5  CNN                   1D convolution over time
Model 6  LSTM                  hidden 40
Model 7  CNN + LSTM            conv -> pool -> LSTM
Model 8  CNN + LSTM + feature attention   the paper's architecture

The attention in Mäkinen et al. is FEATURE attention, not temporal attention:
one scalar weight per input feature, shared across all time steps of that
sample, computed from the sample itself. `FeatureAttention` below implements
exactly that and exposes the weights so they can be inspected.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


class FeatureAttention(nn.Module):
    """One weight per feature, applied across every time step of the sample.

    The sample is summarised over time (mean over the L steps) to give an
    F-vector, a small MLP maps that to F logits, and the resulting weights
    multiply the corresponding feature channel at every time step. This is
    causal at sample level: the summary uses only the input window, which
    contains no information after t.
    """

    def __init__(self, n_features, hidden=None):
        super().__init__()
        h = hidden or max(n_features // 2, 8)
        self.net = nn.Sequential(nn.Linear(n_features, h), nn.Tanh(),
                                 nn.Linear(h, n_features))
        self.last_weights = None

    def forward(self, x):                    # x: [B, L, F]
        s = x.mean(dim=1)                    # [B, F]
        w = torch.softmax(self.net(s), dim=-1) * x.shape[-1]   # mean weight 1
        self.last_weights = w.detach()
        return x * w.unsqueeze(1)


class MLPSnapshot(nn.Module):
    """Model 3: only the last time step."""

    def __init__(self, n_features, hidden=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.LeakyReLU(),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x[:, -1, :]).squeeze(-1)


class MLPHistory(nn.Module):
    """Model 4: the whole window flattened. Dropout because the input is large."""

    def __init__(self, n_features, seq_len, hidden=64, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Dropout(p),
                                 nn.Linear(n_features * seq_len, hidden),
                                 nn.LeakyReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, 40), nn.LeakyReLU(),
                                 nn.Linear(40, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """Model 5: local temporal patterns."""

    def __init__(self, n_features, seq_len, filters=32, k=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, filters, k, padding=k // 2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters, k, padding=k // 2), nn.ReLU(),
            nn.MaxPool1d(2))
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(filters * (seq_len // 4), 40),
                                  nn.ReLU(), nn.Linear(40, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        return self.head(h).squeeze(-1)


class LSTMModel(nn.Module):
    """Model 6: explicit sequence memory, hidden 40 as in the paper."""

    def __init__(self, n_features, hidden=40, p=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, 40),
                                  nn.ReLU(), nn.Linear(40, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


class CNNLSTM(nn.Module):
    """Model 7: conv -> pool -> LSTM, i.e. the paper minus attention."""

    def __init__(self, n_features, filters=32, k=5, hidden=40, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, filters, k, padding=k // 2), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, 40),
                                  nn.ReLU(), nn.Linear(40, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(h)
        return self.head(o[:, -1, :]).squeeze(-1)


class CNNLSTMAttention(nn.Module):
    """Model 8: Mäkinen et al. -- feature attention, then conv/pool/LSTM."""

    def __init__(self, n_features, filters=32, k=5, hidden=40, p=0.5):
        super().__init__()
        self.att = FeatureAttention(n_features)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, filters, k, padding=k // 2), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, 40),
                                  nn.ReLU(), nn.Linear(40, 1))

    def forward(self, x):
        x = self.att(x)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(h)
        return self.head(o[:, -1, :]).squeeze(-1)

    def attention_weights(self):
        return self.att.last_weights


def build(name, n_features, seq_len):
    if name == "mlp_snapshot":
        return MLPSnapshot(n_features)
    if name == "mlp_history":
        return MLPHistory(n_features, seq_len)
    if name == "cnn":
        return CNN1D(n_features, seq_len)
    if name == "lstm":
        return LSTMModel(n_features)
    if name == "cnn_lstm":
        return CNNLSTM(n_features)
    if name == "cnn_lstm_attention":
        return CNNLSTMAttention(n_features)
    raise ValueError("unknown model %s" % name)
