"""Population Stability Index (PSI) against a stored reference profile.

Rule of thumb used in credit risk: PSI < 0.1 stable, 0.1-0.25 moderate shift,
> 0.25 significant shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-4


def _bin_numeric(x: np.ndarray, edges: list[float]) -> np.ndarray:
    # bin i holds edges[i-1] < x <= edges[i]; works for binary features too.
    idx = np.searchsorted(np.asarray(edges), x, side="left")
    return np.bincount(idx, minlength=len(edges) + 1) / max(len(x), 1)


def fit_profile(df: pd.DataFrame, n_bins: int = 10) -> dict:
    profile: dict = {"numeric": {}, "categorical": {}}
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    for col in NUMERIC_FEATURES:
        x = df[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, q)).tolist()
        profile["numeric"][col] = {"edges": edges, "props": _bin_numeric(x, edges).tolist()}
    for col in CATEGORICAL_FEATURES:
        props = df[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = props.to_dict()
    return profile


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(profile: dict, df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col, ref in profile["numeric"].items():
        out[col] = psi(ref["props"], _bin_numeric(df[col].to_numpy(dtype=float), ref["edges"]))
    for col, ref in profile["categorical"].items():
        cats = sorted(set(ref) | set(df[col].astype(str).unique()))
        cur = df[col].astype(str).value_counts(normalize=True)
        out[col] = psi([ref.get(c, 0.0) for c in cats], [cur.get(c, 0.0) for c in cats])
    return out
