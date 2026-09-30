"""Method 2: multi-market CNN. Every market's last 20 s of book is stacked as input channels
(target market first, then the others in a fixed order), so the convolutions can learn patterns
across markets, e.g. the totals book emptying seconds before the moneyline moves.

Input per sample: (K_MAX=10 markets x 8 channels) x 100 steps of 200 ms, plus game state.

    python research/large_jumps/cross_market/m2_cnn.py [--smoke]
"""
import os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K_MAX, SEQ_CH, smoke_flag
from deep import run


class MultiMarketCNN(nn.Module):
    def __init__(self, mu, sd, n_game):
        super().__init__()
        self.register_buffer('mu', mu); self.register_buffer('sd', sd)
        c = K_MAX * SEQ_CH
        self.conv = nn.Sequential(nn.Conv1d(c, 96, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(96, 96, 5, padding=4, dilation=2), nn.ReLU(),
                                  nn.Conv1d(96, 96, 5, padding=8, dilation=4), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192 + n_game, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, X, R, G):                       # X: (B, K, T, C)
        X = (X - self.mu) / self.sd
        X = X * (R >= 0)[:, :, None, None]            # empty market slots stay zero after normalising
        B, K, T, C = X.shape
        h = self.conv(X.permute(0, 1, 3, 2).reshape(B, K * C, T))
        z = torch.cat([h.mean(2), h[:, :, -10:].mean(2), G], 1)
        return self.head(z).squeeze(1)


if __name__ == '__main__':
    run('m2_multimarket_cnn', lambda mu, sd, ng: MultiMarketCNN(mu, sd, ng), smoke_flag(sys.argv))
