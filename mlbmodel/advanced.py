"""Two model families that consistently outperform a plain binary classifier
on low-signal sports data.

1. Residual / offset learning
   A binary classifier spends most of its capacity rediscovering "the better
   team wins". Feeding the pitcher-aware Elo logit in as a fixed offset
   (LightGBM's `init_score`) means the trees only have to model what Elo
   misses, which is a far easier target at this sample size.

2. Run-differential regression
   A 1-0 win and a 12-1 win are the same label to a classifier, but they are
   very different evidence. Regressing the run margin and then mapping it
   through a fitted logistic link uses roughly 3-4x more information per game.
   This is the same reason point-spread models beat moneyline classifiers in
   basketball and football.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class OffsetGBM:
    """LightGBM that learns only the residual on top of a prior probability."""

    def __init__(self, offset_col="elo_sp_logit", shrink=0.85, **kw):
        self.offset_col = offset_col
        self.shrink = shrink
        self.params = dict(
            n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
            min_child_samples=300, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=30.0, verbose=-1,
            n_jobs=4, random_state=0)
        self.params.update(kw)
        self.imp = SimpleImputer(strategy="median")

    def fit_frame(self, frame, feat_cols, y):
        # leagues without starting pitchers fall back to the plain Elo logit
        if self.offset_col not in frame.columns:
            self.offset_col = "elo_logit"
        X = self.imp.fit_transform(frame[feat_cols].to_numpy(dtype=float))
        off = np.nan_to_num(frame[self.offset_col].to_numpy(dtype=float))
        self.m_ = lgb.LGBMClassifier(**self.params)
        self.m_.fit(X, y, init_score=off)
        return self

    def predict_frame(self, frame, feat_cols):
        X = self.imp.transform(frame[feat_cols].to_numpy(dtype=float))
        off = np.nan_to_num(frame[self.offset_col].to_numpy(dtype=float))
        raw = self.m_.predict(X, raw_score=True) + off
        p = 1 / (1 + np.exp(-raw))
        base = 0.5
        return self.shrink * p + (1 - self.shrink) * base


class MarginModel:
    """Regress the run margin, then learn margin -> P(win) on the training set."""

    def __init__(self, kind="ridge", shrink=0.92, **kw):
        self.kind = kind
        self.shrink = shrink
        self.kw = kw
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def fit_frame(self, frame, feat_cols, y):
        X = self.sc.fit_transform(self.imp.fit_transform(
            frame[feat_cols].to_numpy(dtype=float)))
        margin = (frame.home_score - frame.away_score).to_numpy(dtype=float)
        if self.kind == "ridge":
            self.m_ = Ridge(alpha=self.kw.get("alpha", 300.0)).fit(X, margin)
        else:
            self.m_ = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.02, num_leaves=7, max_depth=3,
                min_child_samples=300, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.6, reg_lambda=30.0, verbose=-1,
                n_jobs=4, random_state=0).fit(X, margin)
        # calibrate predicted margin -> win probability on the same data
        mhat = self.m_.predict(X).reshape(-1, 1)
        self.link_ = LogisticRegression(C=1e4, max_iter=1000).fit(mhat, y)
        self.base_ = float(np.mean(y))
        return self

    def predict_frame(self, frame, feat_cols):
        X = self.sc.transform(self.imp.transform(frame[feat_cols].to_numpy(dtype=float)))
        mhat = self.m_.predict(X).reshape(-1, 1)
        p = self.link_.predict_proba(mhat)[:, 1]
        return self.shrink * p + (1 - self.shrink) * self.base_


class MarketBlend:
    """Blend the model with the closing line in logit space.

    Included as a diagnostic, not a strategy: if the optimal weight on the
    model is near zero, the model carries no information the market lacks.
    """

    def __init__(self, inner, w_model=0.5):
        self.inner = inner
        self.w = w_model

    def fit_frame(self, frame, feat_cols, y):
        self.inner.fit_frame(frame, feat_cols, y)
        return self

    def predict_frame(self, frame, feat_cols):
        p = self.inner.predict_frame(frame, feat_cols)
        m = frame["mkt_p_home"].to_numpy(dtype=float)
        z = np.where(np.isfinite(m),
                     self.w * _logit(p) + (1 - self.w) * _logit(np.nan_to_num(m, nan=0.54)),
                     _logit(p))
        return 1 / (1 + np.exp(-z))


FRAME_FIT_MODELS = {
    "gbm_elo_offset": lambda: OffsetGBM(),
    "margin_ridge": lambda: MarginModel("ridge"),
    "margin_gbm": lambda: MarginModel("gbm"),
}
