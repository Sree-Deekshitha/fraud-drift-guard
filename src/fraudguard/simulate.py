"""Synthetic card-transaction stream with controllable drift.

Real fraud data is either private or heavily anonymised (PCA components with no
meaning), which makes it impossible to *inject* a known drift and check that the
monitoring catches it. So the stream is simulated, and every generative
assumption lives in this file.

Three fraud typologies are modelled:

* ``classic_cnp``     - stolen card details used online: high amounts, odd hours,
                        often far from the cardholder, often a new device.
* ``account_takeover``- a compromised account: new device, amount far above the
                        customer's usual spend, older accounts.
* ``card_testing``    - small "test" charges on digital goods / gift cards at high
                        velocity, followed by cash-out purchases. Rare at first; becomes the
                        dominant pattern after ``Scenario.new_pattern_start``.

On top of that, a holiday season shifts legitimate behaviour (more online and
electronics spend, bigger baskets, higher velocity) without changing fraud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CATEGORIES, SEED

LEGIT_CAT_P = np.array([0.27, 0.14, 0.18, 0.20, 0.06, 0.04, 0.03, 0.08])
HOLIDAY_CAT_P = np.array([0.19, 0.10, 0.14, 0.30, 0.11, 0.04, 0.06, 0.06])
CAT_AMOUNT_MULT = dict(
    grocery=1.3, fuel=1.0, restaurants=0.8, online_retail=1.6,
    electronics=4.0, travel=6.0, gift_cards=1.5, digital_goods=0.35,
)
CARD_PRESENT_P = dict(
    grocery=0.92, fuel=0.95, restaurants=0.88, online_retail=0.0,
    electronics=0.45, travel=0.10, gift_cards=0.35, digital_goods=0.0,
)
HOLIDAY_BOOSTED = {"online_retail", "electronics", "gift_cards"}


@dataclass(frozen=True)
class Scenario:
    weekly_volume: int = 20_000
    base_fraud_rate: float = 0.012
    holiday_weeks: tuple[int, int] = (13, 18)  # inclusive
    new_pattern_start: int = 15
    new_pattern_ramp_weeks: int = 3
    card_testing_share_before: float = 0.03
    card_testing_share_after: float = 0.55

    def is_holiday(self, week: int) -> bool:
        return self.holiday_weeks[0] <= week <= self.holiday_weeks[1]

    def pattern_progress(self, week: int) -> float:
        if week < self.new_pattern_start:
            return 0.0
        return min(1.0, (week - self.new_pattern_start + 1) / self.new_pattern_ramp_weeks)

    def card_testing_share(self, week: int) -> float:
        p = self.pattern_progress(week)
        return self.card_testing_share_before + p * (
            self.card_testing_share_after - self.card_testing_share_before
        )

    def fraud_rate(self, week: int) -> float:
        return self.base_fraud_rate * (1.0 + 0.6 * self.pattern_progress(week))


def _hour_daytime(rng, n, mu=14.0, sd=4.0):
    return np.round(rng.normal(mu, sd, n)).astype(int) % 24


def _customer_avg(rng, n):
    return np.exp(rng.normal(3.6, 0.6, n))


def _legit(rng: np.random.Generator, n: int, holiday: bool) -> pd.DataFrame:
    cat = rng.choice(CATEGORIES, size=n, p=HOLIDAY_CAT_P if holiday else LEGIT_CAT_P)
    avg = _customer_avg(rng, n)
    mult = np.vectorize(CAT_AMOUNT_MULT.get)(cat)
    amount = avg * mult * np.exp(rng.normal(0, 0.7, n))
    if holiday:
        amount *= np.where(np.isin(cat, list(HOLIDAY_BOOSTED)), 1.35, 1.0)
    card_present = rng.random(n) < np.vectorize(CARD_PRESENT_P.get)(cat)

    hour = _hour_daytime(rng, n)
    night_owl = rng.random(n) < 0.06
    hour[night_owl] = rng.integers(0, 24, night_owl.sum())

    traveling = rng.random(n) < 0.04
    distance = np.where(card_present, rng.exponential(8, n), rng.exponential(25, n))
    distance = np.where(traveling, rng.exponential(600, n), distance)

    new_device = (~card_present) & (rng.random(n) < 0.07)

    # Power users (subscriptions, in-app purchases, delivery) transact a lot.
    power = rng.random(n) < 0.08
    velocity = np.where(
        power, rng.poisson(9, n) + 2, rng.poisson(2.7 if holiday else 1.5, n) + 1
    )
    age = rng.exponential(900, n) + 1
    newbie = rng.random(n) < 0.10
    age[newbie] = rng.uniform(1, 60, newbie.sum())

    return pd.DataFrame(dict(
        amount=amount, customer_avg=avg, hour=hour, card_present=card_present,
        distance_from_home_km=distance, new_device=new_device,
        txn_count_24h=velocity, account_age_days=age, merchant_category=cat,
        fraud_type="legit",
    ))


def _classic(rng, n):
    cat = rng.choice(
        ["electronics", "travel", "gift_cards", "online_retail", "digital_goods"],
        size=n, p=[0.35, 0.20, 0.15, 0.20, 0.10],
    )
    hour = _hour_daytime(rng, n, sd=5)
    night = rng.random(n) < 0.4
    hour[night] = rng.integers(0, 6, night.sum())
    far = rng.random(n) < 0.55
    return pd.DataFrame(dict(
        amount=np.exp(rng.normal(np.log(320), 0.9, n)),
        customer_avg=_customer_avg(rng, n), hour=hour,
        card_present=np.zeros(n, bool),
        distance_from_home_km=np.where(
            far, np.exp(rng.normal(np.log(450), 1.0, n)), rng.exponential(30, n)
        ),
        new_device=rng.random(n) < 0.6,
        txn_count_24h=rng.poisson(2.5, n) + 1,
        account_age_days=rng.exponential(900, n) + 1,
        merchant_category=cat, fraud_type="classic_cnp",
    ))


def _ato(rng, n):
    avg = _customer_avg(rng, n)
    return pd.DataFrame(dict(
        amount=avg * np.exp(rng.normal(np.log(6), 0.6, n)), customer_avg=avg,
        hour=_hour_daytime(rng, n, mu=15, sd=5),
        card_present=np.zeros(n, bool),
        distance_from_home_km=np.exp(rng.normal(np.log(120), 1.2, n)),
        new_device=rng.random(n) < 0.85,
        txn_count_24h=rng.poisson(2, n) + 1,
        account_age_days=rng.exponential(1500, n) + 180,
        merchant_category=rng.choice(
            ["online_retail", "electronics", "gift_cards", "travel"], size=n,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
        fraud_type="account_takeover",
    ))


def _card_testing(rng, n):
    cashout = rng.random(n) < 0.45
    return pd.DataFrame(dict(
        amount=np.where(cashout, rng.uniform(80, 600, n), rng.uniform(1, 15, n)),
        customer_avg=_customer_avg(rng, n),
        hour=rng.integers(0, 24, n),
        card_present=np.zeros(n, bool),
        distance_from_home_km=rng.exponential(25, n),
        new_device=rng.random(n) < 0.15,
        txn_count_24h=rng.poisson(7, n) + 2,
        account_age_days=rng.exponential(900, n) + 1,
        merchant_category=rng.choice(
            ["digital_goods", "gift_cards", "online_retail"], size=n, p=[0.55, 0.35, 0.10]
        ),
        fraud_type="card_testing",
    ))


def generate_week(week: int, scenario: Scenario = Scenario(), seed: int = SEED) -> pd.DataFrame:
    """Generate one week of transactions. Deterministic per (seed, week)."""
    rng = np.random.default_rng([seed, week])
    n = scenario.weekly_volume
    n_fraud = rng.binomial(n, scenario.fraud_rate(week))

    ct = scenario.card_testing_share(week)
    types = rng.choice(3, size=n_fraud, p=[(1 - ct) * 0.6, (1 - ct) * 0.4, ct])
    parts = [
        _legit(rng, n - n_fraud, scenario.is_holiday(week)),
        _classic(rng, int((types == 0).sum())),
        _ato(rng, int((types == 1).sum())),
        _card_testing(rng, int((types == 2).sum())),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    df["amount"] = df["amount"].clip(0.5, 20_000).round(2)
    df["amount_to_avg_ratio"] = (df["amount"] / df["customer_avg"]).round(3)
    df["card_present"] = df["card_present"].astype(int)
    df["new_device"] = df["new_device"].astype(int)
    df["distance_from_home_km"] = df["distance_from_home_km"].round(1)
    df["account_age_days"] = df["account_age_days"].round(0)
    df["is_fraud"] = (df["fraud_type"] != "legit").astype(int)
    df["week"] = week
    df.insert(0, "txn_id", [f"w{week:02d}-{i:06d}" for i in range(len(df))])
    return df.drop(columns="customer_avg")


def generate_weeks(weeks, scenario: Scenario = Scenario(), seed: int = SEED) -> pd.DataFrame:
    return pd.concat([generate_week(w, scenario, seed) for w in weeks], ignore_index=True)
