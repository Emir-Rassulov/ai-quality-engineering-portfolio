import numpy as np
import pytest

from evaluate import (
    FORBIDDEN_FEATURES,
    PRE_CALL_FEATURES,
    bootstrap_recall_interval,
    build_verdict,
    choose_threshold,
    expected_cost,
    precision_recall,
)
from ml_model import assert_no_forbidden_features, build_model


# --- precision / recall ---------------------------------------------------
def test_precision_and_recall():
    precision, recall = precision_recall(tp=7, fp=3, fn=5)
    assert precision == pytest.approx(0.7)
    assert recall == pytest.approx(7 / 12)


def test_no_failures_detected_returns_zero_not_error():
    assert precision_recall(tp=0, fp=0, fn=5) == (0.0, 0.0)


def test_no_actual_failures_returns_zero_recall():
    assert precision_recall(tp=0, fp=4, fn=0) == (0.0, 0.0)


# --- leakage guard --------------------------------------------------------
def test_pre_call_features_contain_no_forbidden_features():
    assert set(PRE_CALL_FEATURES).isdisjoint(FORBIDDEN_FEATURES)


@pytest.mark.parametrize("leaky", sorted(FORBIDDEN_FEATURES))
def test_guard_rejects_each_forbidden_feature(leaky):
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_forbidden_features(PRE_CALL_FEATURES + [leaky])


def test_model_columns_come_from_the_guarded_feature_list():
    """build_model must use exactly PRE_CALL_FEATURES (one source of truth)."""
    transformers = build_model().named_steps["preprocessing"].transformers
    used = [col for _, _, cols in transformers for col in cols]
    assert sorted(used) == sorted(PRE_CALL_FEATURES)


# --- cost and threshold ---------------------------------------------------
def test_expected_cost():
    assert expected_cost(fp=3, fn=2, cost_fp=1, cost_fn=10) == 23


def test_choose_threshold_picks_lowest_cost():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    assert choose_threshold(y, p, cost_fn=10, cost_fp=1, candidates=[0.2, 0.5, 0.95]) == 0.5


def test_choose_threshold_tie_goes_to_higher_threshold():
    y = np.array([1, 0])
    p = np.array([0.9, 0.1])
    assert choose_threshold(y, p, cost_fn=1, cost_fp=1, candidates=[0.3, 0.5, 0.7]) == 0.7


def test_high_false_negative_cost_lowers_the_threshold():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.4, 0.5, 0.2, 0.1])
    cheap_miss = choose_threshold(y, p, cost_fn=1, cost_fp=1, candidates=[0.3, 0.6])
    costly_miss = choose_threshold(y, p, cost_fn=100, cost_fp=1, candidates=[0.3, 0.6])
    assert costly_miss < cheap_miss


# --- bootstrap ------------------------------------------------------------
def test_bootstrap_interval_is_ordered_and_within_0_1():
    y = np.array([1] * 10 + [0] * 40)
    pred = np.array([1] * 6 + [0] * 4 + [0] * 40)
    low, high = bootstrap_recall_interval(y, pred, iterations=300)
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_is_deterministic():
    y = np.array([1] * 10 + [0] * 40)
    pred = np.array([1] * 5 + [0] * 45)
    assert bootstrap_recall_interval(y, pred, 200) == bootstrap_recall_interval(y, pred, 200)


def test_bootstrap_without_any_failure_raises():
    with pytest.raises(ValueError, match="bootstrap"):
        bootstrap_recall_interval(np.zeros(20, dtype=int), np.zeros(20, dtype=int), 50)


# --- verdict is computed, not hardcoded -----------------------------------
def test_verdict_rejects_weak_model():
    lines = build_verdict(0.10, 0.17, 0.08, cost_model=119, cost_none=120, cost_all=136,
                          interval=(0.0, 0.43), n_failures=12)
    assert "not production-ready" in lines[-1]
    assert "no meaningful saving" in lines[2]


def test_verdict_accepts_strong_model():
    lines = build_verdict(0.60, 0.80, 0.08, cost_model=50, cost_none=120, cost_all=136,
                          interval=(0.6, 0.95), n_failures=12)
    assert lines[-1] == "VERDICT: usable as an alert."


def test_verdict_text_changes_with_the_numbers():
    weak = build_verdict(0.10, 0.17, 0.08, 119, 120, 136, (0.0, 0.43), 12)
    strong = build_verdict(0.60, 0.80, 0.08, 50, 120, 136, (0.6, 0.95), 12)
    assert weak != strong
