"""Weekly production simulation comparing three model-maintenance policies.

Each simulated week:
  1. the current champion scores the week's traffic (we record what *actually*
     happened, using ground truth, for the post-hoc comparison);
  2. the monitor computes feature PSI against the champion's training profile;
  3. chargeback labels for week ``w - LABEL_DELAY_WEEKS`` arrive;
  4. the policy decides whether to retrain. A challenger is trained on recent
     labelled weeks and only promoted if it beats the champion on a held-out
     labelled week (champion/challenger gate). It serves from week ``w + 1``.

Policies:
  * ``static``      - never retrain (the common "deploy and forget" baseline).
  * ``psi_trigger`` - retrain when any feature PSI > 0.25.
  * ``perf_trigger``- retrain when *observed* (label-delayed) recall drops > 8
                      points below validation, or cost per 1k txns rises > 30%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import LABEL_DELAY_WEEKS, TARGET
from .drift import psi_report
from .model import FraudModel, fit_model

PSI_ALERT = 0.25
RECALL_DROP = 0.08
COST_RISE = 1.30
COOLDOWN_WEEKS = 3
TRAIN_WINDOW = 8  # weeks of labelled data for a challenger (see window ablation)


@dataclass
class PolicyRun:
    name: str
    weekly: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.weekly).assign(policy=self.name)


def _should_retrain(policy, champion, psi, observed, week, last_retrain) -> str | None:
    if policy == "static" or week - last_retrain < COOLDOWN_WEEKS:
        return None
    if policy == "psi_trigger":
        feat, value = max(psi.items(), key=lambda kv: kv[1])
        return f"PSI {feat}={value:.2f}" if value > PSI_ALERT else None
    if policy == "perf_trigger" and observed is not None:
        base = champion.val_metrics
        if observed["recall"] < base["recall"] - RECALL_DROP:
            return f"observed recall {observed['recall']:.2f} vs {base['recall']:.2f}"
        if observed["cost_per_1k"] > COST_RISE * base["cost_per_1k"]:
            return f"cost/1k ${observed['cost_per_1k']:.0f} vs ${base['cost_per_1k']:.0f}"
    return None


def run_policy(policy: str, initial: FraudModel, data: pd.DataFrame, prod_weeks: list[int],
               train_window: int = TRAIN_WINDOW) -> PolicyRun:
    run = PolicyRun(policy)
    champion = initial
    last_retrain = -99
    n_versions = 1
    by_week = {w: g for w, g in data.groupby("week")}

    for w in prod_weeks:
        week_df = by_week[w]
        actual = champion.evaluate(week_df)
        psi = psi_report(champion.profile, week_df)

        labelled_week = w - LABEL_DELAY_WEEKS
        observed = champion.evaluate(by_week[labelled_week]) if labelled_week >= prod_weeks[0] else None

        run.weekly.append(dict(
            week=w, model_version=champion.version, **actual,
            psi_max=max(psi.values()), psi_max_feature=max(psi, key=psi.get),
            observed_recall=None if observed is None else observed["recall"],
        ))

        reason = _should_retrain(policy, champion, psi, observed, w, last_retrain)
        if not reason:
            continue
        # Challenger: train on [L-window-1, L-2], validate on L-1, gate on L.
        hold = labelled_week
        train_weeks = list(range(max(0, hold - train_window - 1), hold - 1))
        train = data[data["week"].isin(train_weeks)]
        val = by_week[hold - 1]
        challenger = fit_model(train, val, version=f"v{n_versions + 1}")
        champ_cost = champion.evaluate(by_week[hold])["total_cost"]
        chall_cost = challenger.evaluate(by_week[hold])["total_cost"]
        promoted = chall_cost < champ_cost
        run.events.append(dict(
            week=w, reason=reason, challenger=challenger.version, promoted=promoted,
            champion_holdout_cost=round(champ_cost, 2), challenger_holdout_cost=round(chall_cost, 2),
            train_weeks=train_weeks,
        ))
        last_retrain = w
        if promoted:
            n_versions += 1
            champion = challenger
    return run
