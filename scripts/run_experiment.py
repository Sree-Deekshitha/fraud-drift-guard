"""Offline experiment: rules engine vs logistic regression vs LightGBM.

Split (time-ordered, no shuffling):  train weeks 0-4 | validation 5-6 | test 7
Thresholds are chosen on validation by minimising business cost, then frozen.
Outputs: reports/experiment_results.json, docs/images/*.png, models/production (v1).
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraudguard.config import (
    CATEGORIES, IMAGES_DIR, PRODUCTION_DIR, REPORTS_DIR, TARGET,
)
from fraudguard.costs import best_threshold, cost_breakdown
from fraudguard.model import FEATURES, fit_model, to_frame
from fraudguard.plotting import INK_2, MUTED, SERIES, apply_style, plt
from fraudguard.simulate import generate_weeks

TRAIN, VAL, TEST = [0, 1, 2, 3, 4], [5, 6], [7]


def rules_engine(df: pd.DataFrame) -> np.ndarray:
    """The kind of hand-written rules many teams start with."""
    return (
        ((df["amount"] > 500) & (df["card_present"] == 0))
        | ((df["distance_from_home_km"] > 500) & (df["new_device"] == 1))
        | ((df["hour"] <= 4) & (df["amount"] > 200))
    ).to_numpy()


def logistic_pipeline() -> Pipeline:
    def cyc_hour(x):
        h = np.asarray(x, float)
        return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]

    pre = ColumnTransformer([
        ("log", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]),
         ["amount", "amount_to_avg_ratio", "distance_from_home_km", "account_age_days", "txn_count_24h"]),
        ("hour", FunctionTransformer(cyc_hour), ["hour"]),
        ("bin", "passthrough", ["card_present", "new_device"]),
        ("cat", OneHotEncoder(categories=[CATEGORIES]), ["merchant_category"]),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))])


def main() -> None:
    apply_style()
    for d in (IMAGES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    data = generate_weeks(range(0, 8))
    train, val, test = (data[data.week.isin(w)] for w in (TRAIN, VAL, TEST))
    print(f"Generated {len(data):,} transactions ({data[TARGET].mean():.2%} fraud) "
          f"in {time.perf_counter() - t0:.1f}s")
    print(f"Split  train={len(train):,}  val={len(val):,}  test={len(test):,}\n")

    results, curves = {}, {}
    no_model = cost_breakdown(test[TARGET], np.zeros(len(test), bool), test["amount"])

    flags = rules_engine(test)
    results["Rules engine"] = {"pr_auc": None, "threshold": None,
                               **cost_breakdown(test[TARGET], flags, test["amount"])}

    t = time.perf_counter()
    lr = logistic_pipeline().fit(train[FEATURES], train[TARGET])
    p_val_lr = lr.predict_proba(val[FEATURES])[:, 1]
    thr_lr, _, _ = best_threshold(val[TARGET], p_val_lr, val["amount"])
    p_lr = lr.predict_proba(test[FEATURES])[:, 1]
    results["Logistic regression"] = {
        "pr_auc": average_precision_score(test[TARGET], p_lr), "threshold": thr_lr,
        **cost_breakdown(test[TARGET], p_lr >= thr_lr, test["amount"]),
        "train_seconds": time.perf_counter() - t,
    }
    curves["Logistic regression"] = p_lr

    t = time.perf_counter()
    model = fit_model(train, val, version="v1")
    p_gb = model.predict_proba(test)
    results["LightGBM"] = {
        "pr_auc": average_precision_score(test[TARGET], p_gb), "threshold": model.threshold,
        **model.evaluate(test), "train_seconds": time.perf_counter() - t,
        "best_iteration": model.booster.best_iteration,
    }
    curves["LightGBM"] = p_gb

    # -------- console summary ---------------------------------------------------
    print(f"{'Model':<21}{'PR-AUC':>8}{'Thresh':>8}{'Recall':>8}{'Prec':>8}"
          f"{'Flag%':>7}{'Cost/1k':>10}{'Saved vs none':>15}")
    for name, r in results.items():
        saved = 1 - r["total_cost"] / no_model["total_cost"]
        pr = f"{r['pr_auc']:.3f}" if r["pr_auc"] is not None else "  -  "
        th = f"{r['threshold']:.3f}" if r["threshold"] is not None else "  -  "
        print(f"{name:<21}{pr:>8}{th:>8}{r['recall']:>8.1%}{r['precision']:>8.1%}"
              f"{r['flag_rate']:>7.1%}{r['cost_per_1k']:>10.0f}{saved:>15.1%}")
    print(f"{'No model (approve all)':<21}{'':>8}{'':>8}{0:>8.1%}{'':>8}{0:>7.1%}"
          f"{no_model['cost_per_1k']:>10.0f}{0:>15.1%}")

    test_eval = test.assign(flag=p_gb >= model.threshold)
    by_type = test_eval[test_eval[TARGET] == 1].groupby("fraud_type")["flag"].agg(["mean", "size"])
    print("\nLightGBM recall by fraud type (test week):")
    for ft, row in by_type.iterrows():
        print(f"  {ft:<18} {row['mean']:>6.1%}  (n={int(row['size'])})")

    # -------- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (name, p) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(test[TARGET], p)
        ax.plot(rec, prec, color=SERIES[i], label=f"{name} (PR-AUC {results[name]['pr_auc']:.2f})")
        r = results[name]
        ax.scatter([r["recall"]], [r["precision"]], s=60, color=SERIES[i], edgecolor="white", zorder=3)
    r = results["Rules engine"]
    ax.scatter([r["recall"]], [r["precision"]], s=70, marker="D", color=SERIES[2],
               edgecolor="white", zorder=3, label="Rules engine (fixed)")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="Precision-recall on held-out week 7")
    ax.text(0.02, 0.02, "dots = operating point at the cost-optimal threshold",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06))
    fig.savefig(IMAGES_DIR / "pr_curves.png")
    plt.close(fig)

    p_val = model.predict_proba(val)
    _, grid, costs = best_threshold(val[TARGET], p_val, val["amount"])
    per_1k = costs / len(val) * 1000
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(grid, per_1k, color=SERIES[0])
    ax.axvline(model.threshold, color=INK_2, lw=1, ls="--")
    ax.annotate(f"chosen threshold {model.threshold:.3f}\n${per_1k.min():.0f} per 1k txns",
                xy=(model.threshold, per_1k.min()), xytext=(0.003, per_1k.min() + 0.55 * (per_1k.max() - per_1k.min())),
                color=INK_2, fontsize=9, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xscale("log")
    ax.set(xlabel="Decision threshold (log scale)", ylabel="Cost per 1,000 transactions ($)",
           title="Threshold chosen by business cost, not by 0.5")
    fig.savefig(IMAGES_DIR / "cost_vs_threshold.png")
    plt.close(fig)

    imp = pd.Series(model.booster.feature_importance("gain"), index=model.booster.feature_name())
    imp = (imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.005, y, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
    ax.set(xlabel="Share of total split gain", title="What the model relies on")
    ax.grid(axis="y", visible=False)
    fig.savefig(IMAGES_DIR / "feature_importance.png")
    plt.close(fig)

    # -------- persist ---------------------------------------------------------------
    if PRODUCTION_DIR.exists():
        shutil.rmtree(PRODUCTION_DIR)
    model.save(PRODUCTION_DIR)

    clean = {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                 for kk, vv in v.items()} for k, v in results.items()}
    clean["No model"] = no_model
    (REPORTS_DIR / "experiment_results.json").write_text(json.dumps(
        {"split": {"train": TRAIN, "val": VAL, "test": TEST}, "results": clean,
         "lightgbm_recall_by_fraud_type": by_type["mean"].round(4).to_dict()}, indent=2))

    try:  # optional experiment tracking
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{REPORTS_DIR.parent / 'mlflow.db'}")
        mlflow.set_experiment("fraud-drift-guard")
        with mlflow.start_run(run_name="v1-lightgbm"):
            mlflow.log_params({"threshold": model.threshold, "best_iteration": model.booster.best_iteration,
                               "train_weeks": str(TRAIN), "val_weeks": str(VAL)})
            mlflow.log_metrics({k: float(v) for k, v in results["LightGBM"].items()
                                if isinstance(v, (int, float, np.floating)) and v is not None})
        print("\nLogged run to MLflow (sqlite:///mlflow.db)")
    except Exception as exc:  # pragma: no cover
        print(f"\nMLflow logging skipped: {exc}")

    print(f"Saved model v1 -> {PRODUCTION_DIR.relative_to(PRODUCTION_DIR.parents[1])}")


if __name__ == "__main__":
    main()
