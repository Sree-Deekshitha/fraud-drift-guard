"""Evaluate the LLM copilot on transactions the model actually flagged.

Protocol
  dev  = week 24, used for error analysis and prompt iteration (v1 -> v2)
  test = week 26, untouched until the final prompt was fixed; the headline number
Both weeks are after the new fraud pattern, scored by production model v1.
Per week: stratified sample of flagged cases, up to 15 per true fraud type + 20
false positives. Ground truth = the simulator's fraud_type ("likely_legitimate"
for false positives).

LLM responses are cached in reports/copilot/cases_<prompt>_w<week>.jsonl, so the
report rebuilds without an API key; --refresh deletes the caches and re-calls.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fraudguard.config import FEATURES, IMAGES_DIR, PRODUCTION_DIR, REPORTS_DIR
from fraudguard.copilot import PATTERNS, PRICE_IN, PRICE_OUT, Copilot, grounding_check, load_typical
import lightgbm as lgb

from fraudguard.model import FraudModel, to_frame
from fraudguard.plotting import INK_2, SERIES, apply_style, plt
from fraudguard.simulate import generate_week, generate_weeks

OUT = REPORTS_DIR / "copilot"
RUNS = [("v1", 24, "dev"), ("v2", 24, "dev"), ("v2", 26, "test")]


def rules_pattern(t: dict) -> str:
    """Hand-written analyst heuristic, the baseline the LLM has to beat."""
    if t["txn_count_24h"] >= 6 and t["merchant_category"] in ("digital_goods", "gift_cards"):
        return "card_testing"
    if t["new_device"] and t["amount_to_avg_ratio"] >= 3 and t["account_age_days"] >= 365:
        return "account_takeover"
    if not t["card_present"] and (t["distance_from_home_km"] > 200 or t["hour"] < 6 or t["new_device"]):
        return "classic_cnp"
    return "likely_legitimate"


def eval_set(model: FraudModel, week: int) -> pd.DataFrame:
    df = generate_week(week)
    df = df[model.predict_proba(df) >= model.threshold].copy()
    df["truth"] = df["fraud_type"].replace({"legit": "likely_legitimate"})
    parts = [g.sample(min(len(g), 20 if t == "likely_legitimate" else 15), random_state=0)
             for t, g in df.groupby("truth")]
    return pd.concat(parts).sort_values("txn_id").reset_index(drop=True)


def supervised_ceiling(model: FraudModel, test_week: int) -> dict:
    """How separable are the typologies at all? Train a classifier on ~7.7k *labelled*
    flagged cases from weeks 15-23 and score it on the test week's flagged cases."""
    def flagged(d):
        d = d[model.predict_proba(d) >= model.threshold].copy()
        d["truth"] = d["fraud_type"].replace({"legit": "likely_legitimate"})
        return d
    tr, te = flagged(generate_weeks(range(15, 24))), flagged(generate_week(test_week))
    idx = {c: i for i, c in enumerate(PATTERNS)}
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0)
    clf.fit(to_frame(tr), tr["truth"].map(idx))
    pred = [PATTERNS[i] for i in clf.predict(to_frame(te))]
    per = {t: float((te.truth[te.truth == t] == pd.Series(pred, index=te.index)[te.truth == t]).mean())
           for t in PATTERNS}
    return {"n_train": len(tr), "per_class": per, "macro": float(np.mean(list(per.values())))}


def get_notes(model, prompt, week, cases, records, refresh) -> pd.DataFrame:
    cache = OUT / f"cases_{prompt}_w{week}.jsonl"
    if refresh and cache.exists():
        cache.unlink()
    load = lambda: {json.loads(x)["txn_id"]: json.loads(x) for x in cache.open()} if cache.exists() else {}  # noqa: E731
    cached = load()
    todo = [(t, r) for t, r in zip(cases["txn_id"], records) if t not in cached]
    if todo:
        copilot = Copilot(model, load_typical(PRODUCTION_DIR), prompt=prompt)
        lock = threading.Lock()

        def run(item):
            tid, rec = item
            for attempt in range(2):
                try:
                    note = copilot.explain(rec)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt:
                        print(f"  {tid}: failed ({exc})")
                        return
            with lock, cache.open("a") as f:  # append as we go: never pay twice after a crash
                f.write(json.dumps({"txn_id": tid, **note.to_dict()}) + "\n")

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(run, todo))
        cached = load()
    missing = [t for t in cases["txn_id"] if t not in cached]
    if missing:
        raise SystemExit(f"{len(missing)} cases have no note; rerun to retry them")
    res = pd.DataFrame([cached[t] for t in cases["txn_id"]])
    # always re-check grounding with the current checker
    p = model.val_metrics["precision"]
    facts = (p, p * 100) if prompt == "v2" else ()  # the base rate stated in the v2 prompt
    res["ungrounded"] = [grounding_check(n, c, facts) for n, c in zip(res["note"], res["case"])]
    return res


def score(cases: pd.DataFrame, res: pd.DataFrame, prompt: str, week: int, split: str) -> dict:
    c = cases.assign(llm=[n["suspected_pattern"] for n in res["note"]],
                     action=[n["recommended_action"] for n in res["note"]])
    acc = lambda col, sub: float((sub[col] == sub["truth"]).mean())  # noqa: E731
    fraud, legit = c[c.truth != "likely_legitimate"], c[c.truth == "likely_legitimate"]
    cost = (res.input_tokens * PRICE_IN + res.output_tokens * PRICE_OUT) / 1e6
    return dict(
        prompt=prompt, week=week, split=split, model=res["model"].iloc[0], n=len(c),
        per_class={t: dict(n=int((c.truth == t).sum()), llm=acc("llm", c[c.truth == t]),
                           rules=acc("rules", c[c.truth == t])) for t in PATTERNS},
        # balanced accuracy = mean of per-class accuracy, so the 20 false positives don't dominate
        llm=float(np.mean([acc("llm", c[c.truth == t]) for t in PATTERNS])),
        rules=float(np.mean([acc("rules", c[c.truth == t]) for t in PATTERNS])),
        fraud_called_legit=float((fraud.llm == "likely_legitimate").mean()),
        fraud_approved=float(fraud.action.eq("approve").mean()),
        fp_called_legit=float((legit.llm == "likely_legitimate").mean()),
        grounded=float((res.ungrounded.apply(len) == 0).mean()),
        latency_p50=float(res.latency_s.median()), latency_p95=float(np.percentile(res.latency_s, 95)),
        tokens_in=float(res.input_tokens.mean()), tokens_out=float(res.output_tokens.mean()),
        cost_per_case=float(cost.mean()), cost_total=float(cost.sum()),
        confusion=pd.crosstab(c.truth, c.llm).reindex(index=PATTERNS, columns=PATTERNS, fill_value=0)
        .to_dict(orient="index"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    model = FraudModel.load(PRODUCTION_DIR)

    results, last = [], {}
    for prompt, week, split in RUNS:
        cases = eval_set(model, week)
        records = [{f: (v.item() if hasattr(v, "item") else v) for f, v in r.items()}
                   for r in cases[FEATURES].to_dict("records")]
        cases["rules"] = [rules_pattern(r) for r in records]
        res = get_notes(model, prompt, week, cases, records, args.refresh)
        results.append(score(cases, res, prompt, week, split))
        last[(prompt, week)] = (cases, res)
    ceiling = supervised_ceiling(model, RUNS[-1][1])
    (OUT / "eval_summary.json").write_text(json.dumps({"runs": results, "supervised_ceiling": ceiling}, indent=2))

    # ---- console report ------------------------------------------------------------
    print(f"Copilot eval - flagged transactions after the new fraud pattern, LLM = {results[0]['model']}")
    print("Ground truth = simulator fraud type; 'likely_legitimate' = model false positive\n")
    head = "".join(f"{f'{r['prompt']} wk{r['week']} ({r['split']})':>18}" for r in results)
    print(f"{'Accuracy':<22}{head}{'rules wk26':>12}{'supervised*':>13}")
    test = results[-1]
    for t in PATTERNS:
        row = "".join(f"{r['per_class'][t]['llm']:>18.0%}" for r in results)
        print(f"{t + ' (n=' + str(test['per_class'][t]['n']) + ')':<22}{row}{test['per_class'][t]['rules']:>12.0%}"
              f"{ceiling['per_class'][t]:>13.0%}")
    print(f"{'Overall (balanced)':<22}" + "".join(f"{r['llm']:>18.0%}" for r in results) + f"{test['rules']:>12.0%}"
          f"{ceiling['macro']:>13.0%}")
    print(f"* LightGBM trained on {ceiling['n_train']:,} labelled flagged cases (weeks 15-23): "
          "an upper reference, not a zero-shot method")
    print()
    for label, key in [("Fraud called legitimate", "fraud_called_legit"), ("Fraud -> 'approve'", "fraud_approved"),
                       ("Notes fully grounded", "grounded")]:
        print(f"{label:<22}" + "".join(f"{r[key]:>18.0%}" for r in results))
    print(f"{'Cost per case':<22}" + "".join(f"{'$' + format(r['cost_per_case'], '.4f'):>18}" for r in results))
    print(f"{'Latency p50 / p95':<22}" + "".join(f"{format(r['latency_p50'], '.1f') + 's / ' + format(r['latency_p95'], '.1f') + 's':>18}" for r in results))
    print(f"\nTotal API spend for this evaluation: ${sum(r['cost_total'] for r in results):.2f}")
    print("\nTest-week confusion (rows = truth, cols = LLM v2):")
    print(pd.DataFrame(test["confusion"]).T.to_string())

    # ---- figure: test week, v2 vs rules ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    labels = PATTERNS + ["overall"]
    llm = [test["per_class"][t]["llm"] for t in PATTERNS] + [test["llm"]]
    rules = [test["per_class"][t]["rules"] for t in PATTERNS] + [test["rules"]]
    x = np.arange(len(labels))
    for i, (vals, name) in enumerate([(llm, f"LLM copilot (prompt v2, {test['model']})"), (rules, "Rules heuristic")]):
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=SERIES[i], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [f"{t.replace('_', ' ')}\n(n={test['per_class'][t]['n']})" for t in PATTERNS] + [f"balanced\noverall"])
    ax.set(ylim=(0, 1.15), ylabel="Correct typology",
           title=f"Which fraud pattern is this? Held-out week {test['week']}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(IMAGES_DIR / "copilot_accuracy.png")
    plt.close(fig)

    cases, res = last[("v2", RUNS[-1][1])]
    examples = {}
    for t in PATTERNS:
        idx = cases.index[(cases.truth == t) & (cases.truth == res.note.apply(lambda n: n["suspected_pattern"]))]
        if len(idx):
            i = idx[0]
            examples[t] = {"txn_id": cases.txn_id[i], "case": res.case[i], "note": res.note[i]}
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
