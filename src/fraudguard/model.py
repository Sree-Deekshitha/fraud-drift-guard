"""Model training, scoring and a tiny file-based model registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CATEGORIES, FEATURES, NUMERIC_FEATURES, SEED, TARGET
from .costs import best_threshold, cost_breakdown
from .drift import fit_profile

CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=40,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    metric="average_precision",
    verbose=-1,
    num_threads=2,
)


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Model input as a DataFrame with a fixed categorical dtype."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    X["merchant_category"] = pd.Categorical(df["merchant_category"], categories=CATEGORIES)
    return X[FEATURES]


def to_array(records: list[dict]) -> np.ndarray:
    """Fast path: plain float matrix, category encoded as its code (same as pandas codes)."""
    out = np.empty((len(records), len(FEATURES)), dtype=np.float64)
    for i, r in enumerate(records):
        for j, col in enumerate(NUMERIC_FEATURES):
            out[i, j] = float(r[col])
        out[i, -1] = CAT_INDEX[r["merchant_category"]]
    return out


def train_booster(train: pd.DataFrame, val: pd.DataFrame, seed: int = SEED) -> lgb.Booster:
    dtrain = lgb.Dataset(to_frame(train), train[TARGET], categorical_feature=["merchant_category"])
    dval = lgb.Dataset(to_frame(val), val[TARGET], reference=dtrain)
    return lgb.train(
        {**LGB_PARAMS, "seed": seed},
        dtrain,
        num_boost_round=1500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )


@dataclass
class FraudModel:
    booster: lgb.Booster
    threshold: float
    version: str
    trained_weeks: list[int]
    val_metrics: dict
    profile: dict
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- scoring ---------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(to_frame(df), num_threads=1)

    def predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_threads=1)

    def evaluate(self, df: pd.DataFrame) -> dict:
        p = self.predict_proba(df)
        return cost_breakdown(df[TARGET], p >= self.threshold, df["amount"])

    # -- persistence -------------------------------------------------------------
    def meta(self) -> dict:
        return dict(
            version=self.version, threshold=self.threshold, trained_weeks=self.trained_weeks,
            val_metrics=self.val_metrics, created_at=self.created_at,
            best_iteration=self.booster.best_iteration, features=FEATURES,
        )

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        (path / "meta.json").write_text(json.dumps(self.meta(), indent=2))
        (path / "reference_profile.json").write_text(json.dumps(self.profile))
        return path

    @classmethod
    def load(cls, path: Path) -> "FraudModel":
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            booster=lgb.Booster(model_file=str(path / "model.txt")),
            threshold=meta["threshold"], version=meta["version"],
            trained_weeks=meta["trained_weeks"], val_metrics=meta["val_metrics"],
            profile=json.loads((path / "reference_profile.json").read_text()),
            created_at=meta["created_at"],
        )


def fit_model(train: pd.DataFrame, val: pd.DataFrame, version: str, seed: int = SEED) -> FraudModel:
    """Train, then pick the cost-minimising threshold on the validation window."""
    booster = train_booster(train, val, seed)
    p_val = booster.predict(to_frame(val), num_threads=2)
    thr, _, _ = best_threshold(val[TARGET], p_val, val["amount"])
    metrics = cost_breakdown(val[TARGET], p_val >= thr, val["amount"])
    weeks = sorted(set(train["week"]) | set(val["week"]))
    return FraudModel(
        booster=booster, threshold=thr, version=version,
        trained_weeks=[int(w) for w in weeks], val_metrics=metrics,
        profile=fit_profile(train),
    )
