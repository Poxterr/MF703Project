"""Loading the pairs table without exhausting memory, and passing predictions between notebooks.

Two problems this module solves:

* ``pairs.parquet`` for 2016-2020 is 335 MB on disk but about 3 GB as a pandas frame with all
  35 columns in float64.  ``load_pairs`` reads only the columns the models and the evaluation
  need and stores the numeric ones as float32, which is roughly a fifth of that.

* Each notebook used to write a full copy of the pairs table with its predictions added
  (``pairs_with_baselines.parquet`` -> ``pairs_with_all_models.parquet``), so notebooks had to be
  run in a fixed order by one person.  Now every model writes one small file,
  ``predictions/<model>.parquet`` (the contract key, the date and the hedge ratio), and any
  notebook joins the ones it wants with ``load_predictions``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import FEATURES, KEY, SENTIMENT

ROW_KEY = KEY + ["date"]            # identifies one pair at a given horizon

# Everything the models, the evaluation and the cost simulation read.
MODEL_COLUMNS = list(dict.fromkeys(
    ROW_KEY + ["year", "split", "dte", "S", "dV", "dS", "y", "w", "d_iv",
               "delta_bucket", "ttm_class"] + FEATURES + SENTIMENT))

_FLOAT32 = ["dte", "S", "dV", "dS", "y", "w", "d_iv", "delta_bucket"] + FEATURES + SENTIMENT


def load_pairs(path: str | Path, columns: list[str] | None = None,
               years: list[int] | None = None, float32: bool = True) -> pd.DataFrame:
    """Read the pairs table lean.

    columns : defaults to ``MODEL_COLUMNS``; pass a shorter list for evaluation-only work
    years   : read only these years (row-group filter, so the others never enter memory)
    """
    columns = MODEL_COLUMNS if columns is None else list(columns)
    filters = [("year", "in", list(years))] if years is not None else None
    df = pd.read_parquet(path, columns=columns, filters=filters)
    if float32:
        for c in _FLOAT32:
            if c in df.columns:
                df[c] = df[c].astype("float32")
    return df


def save_predictions(pairs: pd.DataFrame, model_names: list[str], out_dir: str | Path) -> list[Path]:
    """Write ``<out_dir>/<name>.parquet`` for each model: ROW_KEY + ``delta_<name>``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in model_names:
        col = f"delta_{name}"
        if col not in pairs.columns:
            raise KeyError(f"{col} not in the table; call evaluate.add_predictions first")
        path = out_dir / f"{name}.parquet"
        pairs[ROW_KEY + [col]].to_parquet(path, index=False)
        written.append(path)
    return written


def load_predictions(pairs: pd.DataFrame, model_names: list[str], pred_dir: str | Path) -> pd.DataFrame:
    """Join saved predictions onto ``pairs`` by ROW_KEY.  Returns a new frame.

    Raises if a file is missing, or if any row of ``pairs`` has no prediction, which means the
    predictions were made on a different build of the dataset.
    """
    pred_dir = Path(pred_dir)
    out = pairs
    for name in model_names:
        path = pred_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the notebook that fits '{name}' first")
        pred = pd.read_parquet(path)
        pred["type"] = pred["type"].astype(out["type"].dtype)
        out = out.merge(pred, on=ROW_KEY, how="left", validate="one_to_one")
        missing = int(out[f"delta_{name}"].isna().sum())
        if missing:
            raise ValueError(f"{missing:,} rows have no '{name}' prediction: {path} was made on a "
                             f"different build of pairs.parquet. Re-run that model.")
    return out
