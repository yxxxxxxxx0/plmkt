"""Method 3: attention across markets. Each market's last 20 s is encoded by one shared small CNN;
a transformer then attends across the game's markets (role and "is the target" are embedded, empty
slots are masked), and the target market's token plus game state gives the prediction. Works on a
SET of markets, so games with different numbers of markets need no special handling.

    python research/large_jumps/cross_market/m3_attention.py [--smoke]
"""
import os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K_MAX, SEQ_CH, smoke_flag
from deep import run


class MarketAttention(nn.Module):
    def __init__(self, mu, sd, n_game, d=64):
        super().__init__()
        self.register_buffer('mu', mu); self.register_buffer('sd', sd)
        self.enc = nn.Sequential(nn.Conv1d(SEQ_CH, 32, 5, padding=2), nn.ReLU(),
                                 nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
                                 nn.Conv1d(32, d // 2, 5, padding=8, dilation=4), nn.ReLU())
        self.role = nn.Embedding(4, d)                # 0 moneyline, 1 spread, 2 total, 3 empty
        self.target = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.att = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(d + n_game, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, X, R, G):                       # X: (B, K, T, C), R: (B, K) role or -1
        X = (X - self.mu) / self.sd
        B, K, T, C = X.shape
        h = self.enc(X.reshape(B * K, T, C).transpose(1, 2))
        h = torch.cat([h.mean(2), h[:, :, -10:].mean(2)], 1).reshape(B, K, -1)
        empty = R < 0
        tgt = torch.zeros(B, K, dtype=torch.long); tgt[:, 0] = 1
        h = h + self.role(torch.where(empty, torch.full_like(R, 3), R)) + self.target(tgt)
        z = self.att(h, src_key_padding_mask=empty)[:, 0]
        return self.head(torch.cat([z, G], 1)).squeeze(1)


if __name__ == '__main__':
    run('m3_market_attention', lambda mu, sd, ng: MarketAttention(mu, sd, ng), smoke_flag(sys.argv))
