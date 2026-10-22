"""LLM analyst copilot: turns a flagged transaction into a structured case note.

The ML model decides *whether* to review; the copilot helps the analyst decide
*what to do*. It sees only the transaction, the model score and the model's
top drivers, and must answer in a fixed JSON schema (structured outputs). Every number it
writes is then checked against the case data (``grounding_check``) so that
invented details are caught, not trusted.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORIES, FEATURES, NUMERIC_FEATURES
from .model import FraudModel, to_frame

MODEL_ID = os.getenv("FRAUDGUARD_LLM_MODEL", "claude-sonnet-5-5")
# USD per million tokens. Check current pricing before quoting costs.
PRICE_IN, PRICE_OUT = 2.00, 10.00

PATTERNS = ["classic_cnp", "account_takeover", "card_testing", "likely_legitimate"]
ACTIONS = ["block_and_contact_customer", "step_up_authentication", "manual_review", "approve"]
FIELD_LABELS = {
    "amount": "Amount (USD)",
    "amount_to_avg_ratio": "Amount / customer's 30-day average",
    "hour": "Hour of day (0-23, local)",
    "card_present": "Card physically present (1/0)",
    "distance_from_home_km": "Distance from cardholder's home (km)",
    "new_device": "New device for this customer (1/0)",
    "txn_count_24h": "Transactions on this card in last 24h",
    "account_age_days": "Account age (days)",
    "merchant_category": "Merchant category",
}

SYSTEM_V1 = """You are a fraud-operations copilot at a card issuer. A machine-learning model has \
flagged a transaction for review. Write a concise case note that helps a human analyst decide \
what to do.

Fraud typologies you may assign:
- classic_cnp: stolen card details used for a card-not-present purchase; typically unusual \
time, location or device, and higher-value goods.
- account_takeover: an attacker controls an established customer's account; typically a new \
device and spend far above that customer's normal level.
- card_testing: fraudsters probing whether a card works; typically very small charges on \
digital goods or gift cards, many transactions in a short window, sometimes followed by larger \
purchases.
- likely_legitimate: the evidence is better explained by normal customer behaviour.

Rules:
- Use ONLY the facts in CASE DATA. Do not invent customer history, IP addresses, names or \
anything not given. Quote numbers exactly as given.
- The model score is evidence, not a verdict. If the facts look benign, say likely_legitimate.
- Give 1-4 risk factors and 0-3 mitigating factors. Keep the summary under 280 characters,\n  written for a busy analyst."""

_FACTOR = {
    "type": "object",
    "properties": {
        "field": {"type": "string", "enum": FEATURES + ["fraud_probability"]},
        "observation": {"type": "string"},
    },
    "required": ["field", "observation"],
    "additionalProperties": False,
}
# Structured-output schema: the API guarantees the reply parses and matches it.
SCHEMA = {
    "type": "object",
    "properties": {
        "suspected_pattern": {"type": "string", "enum": PATTERNS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "risk_factors": {"type": "array", "items": _FACTOR},
        "mitigating_factors": {"type": "array", "items": _FACTOR},
        "recommended_action": {"type": "string", "enum": ACTIONS},
        "summary": {"type": "string"},
    },
    "required": ["suspected_pattern", "confidence", "risk_factors", "mitigating_factors",
                 "recommended_action", "summary"],
    "additionalProperties": False,
}


# v2: added after error analysis on the dev week (v1 confused classic CNP with account
# takeover 13/15 times, and agreed with the model on most false positives).
SYSTEM_V2 = SYSTEM_V1 + """

How to tell the typologies apart:
- classic_cnp and account_takeover can both involve a new device. Prefer classic_cnp when the \
purchase is far from the cardholder's home or at unusual night hours, on high-value resellable \
goods. Prefer account_takeover when the main anomaly is spend far above this customer's own \
normal level on an established account, while location and time look ordinary.
- Base rate: historically about {precision:.0%} of transactions flagged at this threshold are \
fraud, so a flag alone is weak evidence. A score close to the threshold with no strong, \
specific anomaly is usually likely_legitimate."""

PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


# --------------------------------------------------------------------------- drivers
def load_typical(model_dir: Path) -> list[float]:
    return json.loads((model_dir / "typical.json").read_text())


def drivers(model: FraudModel, txn: dict, typical: list[float], k: int = 4) -> list[dict]:
    """Leave-one-out attribution on the raw score (same method as the web demo)."""
    X = to_frame(pd.DataFrame([txn]))
    rows = [X]
    for i, f in enumerate(FEATURES):
        Y = X.copy()
        Y[f] = CATEGORIES[int(typical[i])] if f == "merchant_category" else typical[i]
        if f == "merchant_category":
            Y[f] = pd.Categorical(Y[f], categories=CATEGORIES)
        rows.append(Y)
    raw = model.booster.predict(pd.concat(rows, ignore_index=True), raw_score=True, num_threads=1)
    deltas = raw[0] - raw[1:]
    order = np.argsort(-np.abs(deltas))[:k]
    return [{"field": FEATURES[i], "effect": "raises risk" if deltas[i] > 0 else "lowers risk",
             "log_odds": round(float(deltas[i]), 2)} for i in order]


def case_data(model: FraudModel, txn: dict, typical: list[float]) -> dict:
    p = float(model.predict_proba(pd.DataFrame([txn]))[0])
    t = {f: txn[f] for f in FEATURES}
    for f in NUMERIC_FEATURES:
        t[f] = round(float(t[f]), 3)
    return {
        "transaction": {FIELD_LABELS[f]: t[f] for f in FEATURES},
        "model": {"fraud_probability": round(p, 4), "review_threshold": model.threshold,
                  "top_drivers_vs_typical_customer": drivers(model, txn, typical)},
    }


# ------------------------------------------------------------------------ grounding
_NUM = re.compile(r"(?<![\w.])-?\$?\d[\d,]*\.?\d*%?")


def _allowed_numbers(case: dict) -> list[float]:
    vals = []
    for v in case["transaction"].values():
        if isinstance(v, (int, float)):
            vals.append(float(v))
    tx = case["transaction"]
    amt, ratio = tx[FIELD_LABELS["amount"]], tx[FIELD_LABELS["amount_to_avg_ratio"]]
    if ratio:
        vals.append(amt / ratio)  # the customer's average, which can be derived
    p = case["model"]["fraud_probability"]
    vals += [p, p * 100, case["model"]["review_threshold"], case["model"]["review_threshold"] * 100]
    for d in case["model"]["top_drivers_vs_typical_customer"]:
        vals += [d["log_odds"], abs(d["log_odds"])]
    vals += [0, 1, 24, 30]  # appear in field definitions ("last 24h", "30-day average")
    return vals


def grounding_check(note: dict, case: dict, extra: tuple[float, ...] = ()) -> list[str]:
    """Return numbers in the note that don't match the case data (or facts given in the prompt)."""
    allowed = _allowed_numbers(case) + list(extra)
    texts = [x["observation"] for x in note["risk_factors"] + note["mitigating_factors"]]
    texts.append(note["summary"])
    bad = []
    for text in texts:
        for m in _NUM.findall(text):
            v = float(m.replace("$", "").replace(",", "").rstrip("%"))
            ok = any(abs(v - a) <= max(0.06, 0.03 * abs(a)) for a in allowed)
            ok = ok or any(abs(round(a) - v) < 1e-9 for a in allowed)  # rounded integers
            if not ok:
                bad.append(m)
    return bad


# --------------------------------------------------------------------------- client
@dataclass
class CaseNote:
    note: dict
    case: dict
    ungrounded: list[str]
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    prompt: str = "v2"

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT) / 1e6

    def to_dict(self) -> dict:
        return dict(note=self.note, case=self.case, ungrounded=self.ungrounded,
                    input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    latency_s=round(self.latency_s, 3), cost_usd=round(self.cost_usd, 6),
                    model=self.model, prompt=self.prompt)


class Copilot:
    def __init__(self, model: FraudModel, typical: list[float], client=None, llm: str = MODEL_ID,
                 prompt: str = "v2"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.model, self.typical, self.client, self.llm = model, typical, client, llm
        self.prompt = prompt
        p = model.val_metrics["precision"]
        self.system = PROMPTS[prompt].format(precision=p)
        self.prompt_facts = (p, p * 100) if prompt == "v2" else ()

    def explain(self, txn: dict) -> CaseNote:
        case = case_data(self.model, txn, self.typical)
        t = time.perf_counter()
        resp = self.client.messages.create(
            model=self.llm, max_tokens=2000, system=self.system,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": "CASE DATA\n" + json.dumps(case, indent=1)}],
        )
        latency = time.perf_counter() - t
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("case note truncated (max_tokens)")
        note = json.loads(next(b.text for b in resp.content if b.type == "text"))
        return CaseNote(note=note, case=case, ungrounded=grounding_check(note, case, self.prompt_facts),
                        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                        latency_s=latency, model=self.llm, prompt=self.prompt)
