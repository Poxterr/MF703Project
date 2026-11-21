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

from .schema import DERIVED, FEATURES


def feature_matrix(pairs: pd.DataFrame, names: list[str]) -> np.ndarray:
    """float32 matrix of the named features; names in ``schema.DERIVED`` are computed on the fly."""
    cols = []
    for n in names:
        if n in pairs.columns:
            cols.append(pairs[n].to_numpy(dtype="float32"))
        elif n in DERIVED:
            cols.append(DERIVED[n](pairs).astype("float32"))
        else:
            raise KeyError(f"feature {n!r} is neither a column of the table nor in schema.DERIVED")
    return np.column_stack(cols)


def hw_scale(pairs: pd.DataFrame) -> np.ndarray:
    """vega / (S sqrt(T)): the factor Hull & White (2017) put in front of their correction."""
    f = lambda c: pairs[c].to_numpy(dtype="float64")
    return (f("vega") / (f("S") * np.sqrt(f("T")))).astype("float32")


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
        # float64 here even when the table is stored as float32: least squares on millions of
        # rows loses digits in single precision
        d = p["delta"].to_numpy(dtype="float64")
        scale = p["vega"].to_numpy(dtype="float64") / (
            p["S"].to_numpy(dtype="float64") * np.sqrt(p["T"].to_numpy(dtype="float64")))
        return np.column_stack([scale, scale * d, scale * d**2])

    def fit(self, pairs):
        for t, p in pairs.groupby("type", observed=True):
            X = self._design(p)
            r = (p["y"].to_numpy(dtype="float64") - p["delta"].to_numpy(dtype="float64"))
            sw = np.sqrt(p["w"].to_numpy(dtype="float64"))
            beta, *_ = np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)
            self.coef_[t] = beta
        return self

    def predict(self, pairs):
        out = pairs["delta"].to_numpy(dtype="float64").copy()
        for t, idx in pairs.groupby("type", observed=True).indices.items():
            p = pairs.iloc[idx]
            out[idx] += self._design(p) @ self.coef_[t]
        return out


# --------------------------------------------------------------------------- #
@dataclass
class GBMResidual(HedgeModel):
    """LightGBM regression of the residual y - delta_BS on the features, weight dS^2.

    ``features`` are the numeric inputs; a call/put indicator is always added.
    Row subsampling needs ``subsample_freq`` > 0 in LightGBM, otherwise ``subsample`` is ignored.
    """
    name: str = "gbm"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    seed: int = 0
    params: dict = field(default_factory=lambda: dict(
        n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1))

    def _X(self, p: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(feature_matrix(p, self.features), columns=self.features, index=p.index)
        X["is_call"] = (p["type"] == "C").astype("float32")
        return X

    def fit(self, pairs, val: pd.DataFrame | None = None):
        import inspect
        import lightgbm as lgb
        self.model_ = lgb.LGBMRegressor(random_state=self.seed, **self.params)
        kw = {}
        if val is not None and len(val):
            Xv, yv = self._X(val), val["y"] - val["delta"]
            # newer LightGBM takes eval_X / eval_y; older versions only eval_set
            if "eval_X" in inspect.signature(self.model_.fit).parameters:
                kw = dict(eval_X=Xv, eval_y=yv)
            else:
                kw = dict(eval_set=[(Xv, yv)])
            kw.update(eval_sample_weight=[val["w"]], callbacks=[lgb.early_stopping(50, verbose=False)])
        self.model_.fit(self._X(pairs), pairs["y"] - pairs["delta"], sample_weight=pairs["w"], **kw)
        return self

    def predict(self, pairs):
        return pairs["delta"].to_numpy(dtype="float64") + self.model_.predict(self._X(pairs))

    def importance(self) -> pd.Series:
        """Gain-based feature importance, largest first."""
        b = self.model_.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
@dataclass
class NeuralHedger(HedgeModel):
    """Feed-forward network trained on the weighted one-day hedging error.

    residual=True  -> delta = delta_BS + f(x)   (Qiao & Wan 2024, "-BS" models)
    residual=False -> delta = f(x)              (direct learning)

    Architecture follows the paper: hidden layers of 128 units, sigmoid activations,
    Adam at lr 1e-4, batch 1024, early stopping on the validation loss.
    Inputs are standardised with training-set mean and std.

    Two options that matter out of sample (see notebook 03):

    output_scale="hw"  the network output g(x) is multiplied by vega / (S sqrt(T)), so
                       delta = delta_BS + g(x) * vega / (S sqrt(T)).  Hull-White is the special case
                       g = a + b*delta + c*delta^2; the network generalises the bracket.  The
                       correction then scales with vega automatically, as theory says it should.
    clip_inputs=k      standardised inputs are clipped to [-k, k], so a year whose inputs lie far
                       outside the training range is treated as the edge of that range rather than
                       extrapolated.
    features           names of table columns or of ``schema.DERIVED`` (scale-free) features.
    label              overrides the default name ("nn_resid" / "nn_direct") in tables.
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
    verbose: bool = False            # print train / validation loss after every epoch
    eval_batch: int = 200_000        # rows per forward pass when scoring; keeps memory flat on millions of rows
    output_scale: str = "none"       # "none" or "hw" (residual models only)
    clip_inputs: float | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or ("nn_resid" if self.residual else "nn_direct")

    def __post_init__(self):
        if self.output_scale not in ("none", "hw"):
            raise ValueError("output_scale must be 'none' or 'hw'")
        if self.output_scale == "hw" and not self.residual:
            raise ValueError("output_scale='hw' only applies to residual models")

    # -- helpers ---------------------------------------------------------------
    def _X(self, p: pd.DataFrame) -> np.ndarray:
        X = (feature_matrix(p, self.features) - self.mu_) / self.sd_
        if self.clip_inputs is not None:
            X = np.clip(X, -self.clip_inputs, self.clip_inputs)
        return X.astype(np.float32)

    def _scale(self, p: pd.DataFrame) -> np.ndarray:
        """Multiplier applied to the raw network output (1 unless output_scale='hw')."""
        return hw_scale(p) if self.output_scale == "hw" else np.ones(len(p), dtype=np.float32)

    def _forward(self, X):
        """Network output for a tensor of standardised inputs, computed in chunks without gradients."""
        import torch
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.eval_batch):
                out.append(self.net_(X[i:i + self.eval_batch]).squeeze(1))
        return torch.cat(out) if out else torch.empty(0)

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
        X_raw = feature_matrix(pairs, self.features).astype(np.float64)
        self.mu_ = X_raw.mean(0).astype(np.float32)
        self.sd_ = (X_raw.std(0) + 1e-8).astype(np.float32)
        del X_raw

        def tensors(p):
            X = torch.tensor(self._X(p))
            dV = torch.tensor(p["dV"].to_numpy(dtype=np.float32))
            # the output multiplier is folded into dS: (dV - (dbs + g*k) dS) = dV - dbs*dS - g*(k*dS)
            dS = torch.tensor(p["dS"].to_numpy(dtype=np.float32))
            k = torch.tensor(self._scale(p))
            dbs = torch.tensor(p["delta"].to_numpy(dtype=np.float32))
            return X, dV, dS, dbs, k

        Xtr, dVtr, dStr, dbtr, ktr = tensors(pairs)
        self.net_ = self._build(Xtr.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        def loss_fn(X, dV, dS, dbs, k):
            f = self.net_(X).squeeze(1) * k
            delta = dbs + f if self.residual else f
            return ((dV - delta * dS) ** 2).mean()

        def full_loss(X, dV, dS, dbs, k):            # same loss over a whole set, chunked, no gradients
            f = self._forward(X) * k
            delta = dbs + f if self.residual else f
            return ((dV - delta * dS) ** 2).mean().item()

        has_val = val is not None and len(val) > 0
        if has_val:
            Xv, dVv, dSv, dbv, kv = tensors(val)
        best, best_state, bad = np.inf, None, 0
        n = len(Xtr)
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(Xtr[idx], dVtr[idx], dStr[idx], dbtr[idx], ktr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            tr = full_loss(Xtr, dVtr, dStr, dbtr, ktr)
            vl = full_loss(Xv, dVv, dSv, dbv, kv) if has_val else tr
            self.history_.append((epoch, tr, vl))
            if self.verbose:
                print(f"{self.name} epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}", flush=True)
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
        f = self._forward(torch.tensor(self._X(pairs))).numpy() * self._scale(pairs)
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
