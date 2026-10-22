"""20 production weeks with a holiday shift (weeks 13-18) and a new fraud
pattern (from week 15). Compares static / PSI-triggered / performance-triggered
maintenance policies. Outputs reports/monitoring_*.json/csv + timeline figure.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from fraudguard.config import IMAGES_DIR, PRODUCTION_DIR, REPORTS_DIR
from fraudguard.model import FraudModel
from fraudguard.monitor import PSI_ALERT, TRAIN_WINDOW, run_policy
from fraudguard.plotting import CRITICAL, INK_2, MUTED, SERIES, apply_style, plt
from fraudguard.simulate import Scenario, generate_weeks

PROD_WEEKS = list(range(8, 28))
STEADY_FROM = 22  # holiday over, new pattern fully ramped, retrained models settled
POLICIES = ["static", "psi_trigger", "perf_trigger"]
LABELS = {"static": "Static (never retrain)", "psi_trigger": "PSI-triggered",
          "perf_trigger": "Performance-triggered"}


def main() -> None:
    apply_style()
    scenario = Scenario()
    t0 = time.perf_counter()
    data = generate_weeks(range(0, 28), scenario)
    v1 = FraudModel.load(PRODUCTION_DIR)
    print(f"Simulating {len(PROD_WEEKS)} production weeks ({len(data):,} txns). "
          f"Holiday shift: weeks {scenario.holiday_weeks[0]}-{scenario.holiday_weeks[1]}; "
          f"new fraud pattern from week {scenario.new_pattern_start}.\n")

    runs = {p: run_policy(p, v1, data, PROD_WEEKS) for p in POLICIES}
    weekly = pd.concat([r.frame() for r in runs.values()], ignore_index=True)

    for p, r in runs.items():
        print(f"[{LABELS[p]}]")
        if not r.events:
            print("  no retraining")
        for e in r.events:
            verdict = "PROMOTED" if e["promoted"] else "rejected"
            print(f"  week {e['week']:>2}: {e['reason']:<38} -> {e['challenger']} {verdict} "
                  f"(holdout cost ${e['challenger_holdout_cost']:,.0f} vs champion ${e['champion_holdout_cost']:,.0f})")
    print()

    steady = weekly[weekly.week >= STEADY_FROM]
    summary = []
    for p in POLICIES:
        w, sw = weekly[weekly.policy == p], steady[steady.policy == p]
        summary.append(dict(
            policy=LABELS[p], total_cost=w.total_cost.sum(), fraud_loss=w.fraud_loss.sum(),
            review_cost=w.review_cost.sum(), mean_recall=w.recall.mean(),
            steady_cost=sw.total_cost.sum(), steady_recall=sw.recall.mean(),
            retrains=len(runs[p].events), promotions=sum(e["promoted"] for e in runs[p].events),
        ))
    s = pd.DataFrame(summary)
    s["total_vs_static"] = s.total_cost / s.total_cost.iloc[0] - 1
    s["steady_vs_static"] = s.steady_cost / s.steady_cost.iloc[0] - 1
    print(f"{'':<24}{'---- all 20 weeks ----':^30}{'-- steady state (wk 22-27) --':^32}")
    print(f"{'Policy':<24}{'Cost':>10}{'vs static':>11}{'Recall':>9}{'Cost':>12}{'vs static':>11}{'Recall':>9}{'Retrains':>10}")
    for _, r in s.iterrows():
        print(f"{r.policy:<24}{r.total_cost:>10,.0f}{r.total_vs_static:>+11.1%}{r.mean_recall:>9.1%}"
              f"{r.steady_cost:>12,.0f}{r.steady_vs_static:>+11.1%}{r.steady_recall:>9.1%}"
              f"{r.retrains:>6} ({r.promotions})")

    print(f"\nAblation - performance-triggered policy, challenger training window (default {TRAIN_WINDOW}):")
    ablation = []
    for win in (3, 5, 8, 11):
        f = run_policy("perf_trigger", v1, data, PROD_WEEKS, train_window=win).frame()
        ablation.append(dict(window=win, total_cost=f.total_cost.sum(),
                             steady_cost=f[f.week >= STEADY_FROM].total_cost.sum(),
                             steady_recall=f[f.week >= STEADY_FROM].recall.mean()))
        a = ablation[-1]
        print(f"  {win:>2} weeks   total ${a['total_cost']:>9,.0f}   steady ${a['steady_cost']:>8,.0f}"
              f"   steady recall {a['steady_recall']:.1%}")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")

    # -------- figure ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.2, 1.2], hspace=0.28))
    for ax in axes:
        ax.axvspan(scenario.holiday_weeks[0] - 0.5, scenario.holiday_weeks[1] + 0.5,
                   color="#f0efec", zorder=0, lw=0)
        ax.axvline(scenario.new_pattern_start - 0.5, color=CRITICAL, lw=1, ls=":")
    st = weekly[weekly.policy == "static"]
    axes[0].plot(st.week, st.psi_max, color=SERIES[0], marker="o", ms=4)
    axes[0].axhline(PSI_ALERT, color=MUTED, lw=1, ls="--")
    axes[0].text(PROD_WEEKS[0], PSI_ALERT + 0.02, f"alert {PSI_ALERT}", color=MUTED, fontsize=8)
    axes[0].set(ylabel="Max feature PSI", title="Drift signal vs. what actually matters", ylim=(-0.02, 0.62))
    axes[0].text(scenario.holiday_weeks[0] - 0.35, 0.12, "holiday\nseason", color=INK_2, fontsize=8)
    axes[0].text(scenario.new_pattern_start - 0.35, 0.12, "new fraud\npattern", color=CRITICAL, fontsize=8)

    for i, p in enumerate(POLICIES):
        w = weekly[weekly.policy == p]
        axes[1].plot(w.week, w.recall, color=SERIES[i], marker="o", ms=4, label=LABELS[p])
        axes[2].plot(w.week, w.cost_per_1k, color=SERIES[i], marker="o", ms=4, label=LABELS[p])
        for e in runs[p].events:
            if e["promoted"]:
                y = w.loc[w.week == e["week"] + 1, "recall"]
                if len(y):
                    axes[1].scatter([e["week"] + 1], y, s=110, facecolor="none",
                                    edgecolor=SERIES[i], lw=1.6, zorder=4)
    axes[1].set(ylabel="Weekly recall")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", ms=10, mfc="none", mec=MUTED, mew=1.6))
    labels.append("new model goes live")
    axes[1].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    axes[2].set(ylabel="Cost per 1k txns ($)", xlabel="Week")
    axes[2].set_xticks(PROD_WEEKS)
    fig.savefig(IMAGES_DIR / "monitoring_timeline.png")
    plt.close(fig)

    weekly.to_csv(REPORTS_DIR / "monitoring_weekly.csv", index=False)
    (REPORTS_DIR / "monitoring_summary.json").write_text(json.dumps(
        {"summary": s.to_dict(orient="records"), "window_ablation": ablation,
         "events": {p: r.events for p, r in runs.items()}}, indent=2, default=float))


if __name__ == "__main__":
    main()
