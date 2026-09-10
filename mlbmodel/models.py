"""Model zoo.

Every entry is a factory returning an object with `fit(X, y)` and
`predict_proba(X)[:, 1]`. They are deliberately small and heavily
regularised: with ~2400 games a season and a signal-to-noise ratio where
the *market itself* only reaches ~58% accuracy, anything with real capacity
memorises noise.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb


def _pipe(clf, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


class EloBaseline:
    """Not fitted -- reads the Elo win probability straight out of the frame."""

    def __init__(self, col="elo_prob_home"):
        self.col = col

    def fit(self, X, y, frame=None):
        return self

    def predict_proba_frame(self, frame):
        return frame[self.col].to_numpy()


class MarketBaseline:
    def fit(self, X, y, frame=None):
        return self

    def predict_proba_frame(self, frame):
        return frame["mkt_p_home"].to_numpy()


class ShrunkLogit:
    """Logistic regression whose output is shrunk toward the base rate.

    Pulling predictions toward the league-wide home win rate is the single
    cheapest defence against an over-confident model on noisy sports data.
    """

    def __init__(self, C=0.05, shrink=0.90, l1_ratio=0.0):
        # sklearn >=1.8 expresses the penalty purely through l1_ratio:
        # 0 = ridge, 1 = lasso, in between = elastic net.
        solver = "saga" if l1_ratio > 0 else "lbfgs"
        self.pipe = _pipe(LogisticRegression(
            C=C, l1_ratio=l1_ratio, solver=solver, max_iter=4000))
        self.shrink = shrink

    def fit(self, X, y):
        self.base_ = float(np.mean(y))
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.pipe.predict_proba(X)[:, 1]
        return self.shrink * p + (1 - self.shrink) * self.base_

    @property
    def named_steps(self):
        return self.pipe.named_steps


def make_models(n_features: int) -> dict:
    return {
        "logit_l2": lambda: ShrunkLogit(C=0.03, shrink=0.92, l1_ratio=0.0),
        "logit_enet": lambda: ShrunkLogit(C=0.08, shrink=0.92, l1_ratio=0.5),
        "lightgbm": lambda: _pipe(lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.015, num_leaves=7, max_depth=3,
            min_child_samples=250, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=20.0, reg_alpha=1.0,
            verbose=-1, n_jobs=4, random_state=0), scale=False),
        "xgboost": lambda: _pipe(xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.015, max_depth=3,
            min_child_weight=60, subsample=0.7, colsample_bytree=0.6,
            reg_lambda=20.0, reg_alpha=1.0, eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=0), scale=False),
        "random_forest": lambda: _pipe(RandomForestClassifier(
            n_estimators=600, max_depth=7, min_samples_leaf=120,
            max_features="sqrt", n_jobs=4, random_state=0), scale=False),
        "extra_trees": lambda: _pipe(ExtraTreesClassifier(
            n_estimators=600, max_depth=8, min_samples_leaf=120,
            max_features="sqrt", n_jobs=4, random_state=0), scale=False),
        "mlp": lambda: _pipe(MLPClassifier(
            hidden_layer_sizes=(16,), alpha=3.0, learning_rate_init=3e-3,
            max_iter=600, early_stopping=True, n_iter_no_change=25,
            random_state=0)),
        "calibrated_gbm": lambda: CalibratedClassifierCV(
            _pipe(lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.015, num_leaves=7, max_depth=3,
                min_child_samples=250, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.6, reg_lambda=20.0, verbose=-1,
                n_jobs=4, random_state=0), scale=False),
            method="sigmoid", cv=4),
    }


FRAME_MODELS = {
    "elo": lambda: EloBaseline("elo_prob_home"),
    "elo_sp": lambda: EloBaseline("elo_sp_prob_home"),
    "market": lambda: MarketBaseline(),
}
