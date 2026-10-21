"""Integration tests on the real cleaned dataset (skipped if it was not generated)."""
import pytest

from baseline import run_baseline
from ml_testing import needs_data
from evaluate import run_evaluation
from leakage_demo import run_leakage_demo
from ml_data import PRE_CALL_FEATURES, TARGET, load_data, split_data

pytestmark = needs_data

# Floors sit below today's numbers (test recall 0.167, cv recall mean 0.40), so
# normal noise passes and a real regression (for example a wrong split) fails.
MIN_CV_RECALL = 0.25
MIN_LEAKAGE_GAP = 0.5


def test_split_is_deterministic():
    df = load_data()
    first = split_data(df)
    second = split_data(df)
    assert first[1].index.tolist() == second[1].index.tolist()


def test_train_and_test_rows_are_disjoint_and_complete():
    df = load_data()
    X_train, X_test, _, _ = split_data(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(df)


def test_split_is_stratified():
    df = load_data()
    _, _, y_train, y_test = split_data(df)
    assert int(y_test.sum()) == 12
    assert int(y_train.sum()) == 36


def test_split_uses_only_pre_call_features():
    X_train, _, _, _ = split_data(load_data())
    assert list(X_train.columns) == PRE_CALL_FEATURES


def test_metrics_are_deterministic_across_runs():
    a = run_evaluation()
    b = run_evaluation()
    for key in ("threshold", "tp", "fp", "fn", "tn", "precision", "recall", "recall_interval"):
        assert a[key] == b[key]


def test_confusion_matrix_adds_up_to_the_test_set():
    r = run_evaluation()
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == r["test_rows"]
    assert r["tp"] + r["fn"] == r["test_failures"] == 12


def test_cv_recall_stays_above_floor():
    assert run_evaluation()["cv_recall"].mean() >= MIN_CV_RECALL


def test_model_recall_is_above_baseline_recall():
    assert run_evaluation()["recall"] > run_baseline()["recall"]


def test_leakage_inflates_scores():
    r = run_leakage_demo()
    honest_cv_recall = r["run2"][3]
    leaky_cv_recall = r["run1"][3]
    assert leaky_cv_recall - honest_cv_recall >= MIN_LEAKAGE_GAP


def test_no_duplicate_prompts_so_random_split_is_safe():
    assert load_data()["prompt"].duplicated().sum() == 0


def test_target_is_binary_and_matches_dataset():
    df = load_data()
    assert set(df[TARGET].unique()) == {0, 1}
