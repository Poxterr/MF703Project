"""Stage 3: hedging models with a common interface.

Every model implements::

    model.fit(train_pairs)          # train_pairs is a slice of the pairs table
    delta_hat = model.predict(pairs)   # np.ndarray of hedge ratios, one per row

The loss for all of them is the mean squared one-day hedging error

    mean( (dV - delta * dS)^2 ) = mean( w * (y - delta)^2 ),   w = dS^2,  y = dV/dS

so the regression-type models are weighted least squares on y.

Residual models predict f(x) and return delta_BS + f(x); direct models predict delta itself.

Models
------
BlackScholes      the benchmark: returns the file's implied delta
HullWhite         Hull & White (2017): delta_BS + (a + b d + c d^2) * vega / (S sqrt(T)), WLS
GBMResidual       LightGBM on the residual y - delta_BS (non-neural ML baseline)
NeuralHedger      feed-forward net (PyTorch); residual=True is Qiao & Wan's method, False learns delta directly
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import FEATURES


class HedgeModel:
    name: str = "base"

    def fit(self, pairs: pd.DataFrame) -> "HedgeModel":
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class BlackScholes(HedgeModel):
    """Benchmark: the implied Black-Scholes delta from the data file."""
    name = "bs"

    def predict(self, pairs):
        return pairs["delta"].to_numpy()


# --------------------------------------------------------------------------- #
class HullWhite(HedgeModel):
    """Hull & White (2017) minimum-variance delta.

    delta_MV = delta_BS + (a + b*delta_BS + c*delta_BS^2) * vega_BS / (S * sqrt(T))

    Fitted by weighted least squares with weight dS^2 on the residual y - delta_BS.
    Calls and puts are fitted separately (different coefficients), as in the paper.
    """
    name = "hw"

    def __init__(self):
        self.coef_: dict[str, np.ndarray] = {}

    @staticmethod
    def _design(p: pd.DataFrame) -> np.ndarray:
        d = p["delta"].to_numpy()
        scale = p["vega"].to_numpy() / (p["S"].to_numpy() * np.sqrt(p["T"].to_numpy()))
        return np.column_stack([scale, scale * d, scale * d**2])

    def fit(self, pairs):
        for t, p in pairs.groupby("type"):
            X = self._design(p)
            r = (p["y"] - p["delta"]).to_numpy()
            sw = np.sqrt(p["w"].to_numpy())
            beta, *_ = np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)
            self.coef_[t] = beta
        return self

    def predict(self, pairs):
        out = pairs["delta"].to_numpy().copy()
        for t, idx in pairs.groupby("type").indices.items():
            p = pairs.iloc[idx]
            out[idx] += self._design(p) @ self.coef_[t]
        return out


# --------------------------------------------------------------------------- #
@dataclass
class GBMResidual(HedgeModel):
    """LightGBM regression of the residual y - delta_BS on the features, weight dS^2."""
    name: str = "gbm"
    features: list[str] = field(default_factory=lambda: FEATURES + ["is_call"])
    params: dict = field(default_factory=lambda: dict(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8, verbose=-1))

    def _X(self, p: pd.DataFrame) -> pd.DataFrame:
        X = p[FEATURES].copy()
        X["is_call"] = (p["type"] == "C").astype(float)
        return X

    def fit(self, pairs, val: pd.DataFrame | None = None):
        import lightgbm as lgb
        self.model_ = lgb.LGBMRegressor(**self.params)
        kw = {}
        if val is not None:
            kw = dict(eval_set=[(self._X(val), (val["y"] - val["delta"]))],
                      eval_sample_weight=[val["w"]],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        self.model_.fit(self._X(pairs), pairs["y"] - pairs["delta"], sample_weight=pairs["w"], **kw)
        return self

    def predict(self, pairs):
        return pairs["delta"].to_numpy() + self.model_.predict(self._X(pairs))


# --------------------------------------------------------------------------- #
@dataclass
class NeuralHedger(HedgeModel):
    """Feed-forward network trained on the weighted one-day hedging error.

    residual=True  -> delta = delta_BS + f(x)   (Qiao & Wan 2024, "-BS" models)
    residual=False -> delta = f(x)              (direct learning)

    Architecture follows the paper: hidden layers of 128 units, sigmoid activations,
    Adam at lr 1e-4, batch 1024, early stopping on the validation loss.
    Inputs are standardised with training-set mean and std.
    """
    residual: bool = True
    features: list[str] = field(default_factory=lambda: FEATURES)
    hidden: tuple[int, ...] = (128, 128, 128)
    lr: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 10
    seed: int = 0
    device: str = "cpu"

    @property
    def name(self) -> str:
        return "nn_resid" if self.residual else "nn_direct"

    # -- helpers ---------------------------------------------------------------
    def _X(self, p: pd.DataFrame) -> np.ndarray:
        X = p[self.features].to_numpy(dtype=np.float32)
        return (X - self.mu_) / self.sd_

    def _build(self, n_in: int):
        import torch.nn as nn
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.Sigmoid()]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    # -- fit / predict ---------------------------------------------------------
    def fit(self, pairs, val: pd.DataFrame | None = None):
        import torch
        torch.manual_seed(self.seed)
        X_raw = pairs[self.features].to_numpy(dtype=np.float32)
        self.mu_, self.sd_ = X_raw.mean(0), X_raw.std(0) + 1e-8

        def tensors(p):
            X = torch.tensor(self._X(p))
            dV = torch.tensor(p["dV"].to_numpy(dtype=np.float32))
            dS = torch.tensor(p["dS"].to_numpy(dtype=np.float32))
            dbs = torch.tensor(p["delta"].to_numpy(dtype=np.float32))
            return X, dV, dS, dbs

        Xtr, dVtr, dStr, dbtr = tensors(pairs)
        self.net_ = self._build(Xtr.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        def loss_fn(X, dV, dS, dbs):
            f = self.net_(X).squeeze(1)
            delta = dbs + f if self.residual else f
            return ((dV - delta * dS) ** 2).mean()

        has_val = val is not None and len(val) > 0
        if has_val:
            Xv, dVv, dSv, dbv = tensors(val)
        best, best_state, bad = np.inf, None, 0
        n = len(Xtr)
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(Xtr[idx], dVtr[idx], dStr[idx], dbtr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                tr = loss_fn(Xtr, dVtr, dStr, dbtr).item()
                vl = loss_fn(Xv, dVv, dSv, dbv).item() if has_val else tr
            self.history_.append((epoch, tr, vl))
            if vl < best - 1e-9:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, pairs):
        import torch
        self.net_.eval()
        with torch.no_grad():
            f = self.net_(torch.tensor(self._X(pairs))).squeeze(1).numpy()
        return pairs["delta"].to_numpy() + f if self.residual else f

    def correction_surface(self, moneyness, T, template_row: pd.Series) -> np.ndarray:
        """Evaluate f(x) on a grid for interpretability plots (residual models only).

        Other features are held at ``template_row`` values; delta is recomputed from
        moneyness and T with the template's iv via Black-Scholes.
        """
        from .bs import bs_delta
        M, TT = np.meshgrid(moneyness, T)
        rows = pd.DataFrame({"moneyness": M.ravel(), "T": TT.ravel()})
        for c in self.features:
            if c not in rows:
                rows[c] = template_row[c]
        rows["delta"] = bs_delta(1.0, 1.0 / rows["moneyness"], rows["T"], 0.0, rows["iv"], "C")
        import torch
        with torch.no_grad():
            f = self.net_(torch.tensor(self._X(rows))).squeeze(1).numpy()
        return f.reshape(M.shape)
