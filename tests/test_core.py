import numpy as np
import pandas as pd

from fraudguard.config import CATEGORIES, FEATURES
from fraudguard.costs import best_threshold, cost_breakdown
from fraudguard.drift import fit_profile, psi_report
from fraudguard.model import FraudModel, to_array
from fraudguard.simulate import Scenario, generate_week

from .conftest import SMALL


def test_simulation_is_deterministic_and_valid():
    a, b = generate_week(3, SMALL), generate_week(3, SMALL)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURES) <= set(a.columns)
    assert a["merchant_category"].isin(CATEGORIES).all()
    assert a["hour"].between(0, 23).all()
    assert 0.015 < a["is_fraud"].mean() < 0.05


def test_scenario_drift_schedule():
    s = Scenario()
    assert s.card_testing_share(s.new_pattern_start - 1) == s.card_testing_share_before
    assert s.card_testing_share(40) == s.card_testing_share_after
    assert s.is_holiday(s.holiday_weeks[0]) and not s.is_holiday(s.holiday_weeks[1] + 1)


def test_cost_breakdown_counts_missed_fraud_and_reviews():
    y = np.array([1, 1, 0, 0])
    flagged = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 10.0, 10.0])
    c = cost_breakdown(y, flagged, amount)
    assert c["fraud_loss"] == 50 + 25  # missed fraud + chargeback fee
    assert c["review_cost"] == 2 * 12
    assert c["recall"] == 0.5 and c["precision"] == 0.5


def test_best_threshold_separates_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.r_[np.full(90, 0.1), np.full(10, 0.9)]
    thr, _, costs = best_threshold(y, scores, np.full(100, 200.0))
    assert 0.1 < thr <= 0.9
    assert costs.min() == 10 * 12  # only review cost of the 10 true frauds


def test_psi_flags_shift_but_not_same_distribution():
    ref = generate_week(0)
    profile = fit_profile(ref)
    same = psi_report(profile, generate_week(1))
    assert max(same.values()) < 0.05
    shifted = generate_week(2).assign(txn_count_24h=lambda d: d.txn_count_24h * 3)
    assert psi_report(profile, shifted)["txn_count_24h"] > 0.25


def test_fast_scoring_path_matches_pandas_path(trained, small_data):
    model, _ = trained
    sample = small_data.sample(500, random_state=0)
    slow = model.predict_proba(sample)
    fast = model.predict_proba_array(to_array(sample[FEATURES].to_dict("records")))
    np.testing.assert_allclose(slow, fast, rtol=1e-9)


def test_model_roundtrip(trained, small_data):
    model, path = trained
    loaded = FraudModel.load(path)
    sample = small_data.head(200)
    np.testing.assert_allclose(model.predict_proba(sample), loaded.predict_proba(sample))
    assert loaded.threshold == model.threshold and loaded.version == "test"
