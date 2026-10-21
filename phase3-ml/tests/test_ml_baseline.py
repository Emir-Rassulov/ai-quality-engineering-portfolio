import pytest

from baseline import run_baseline
from ml_testing import needs_data

pytestmark = needs_data


def test_dataset_has_expected_rows_and_failures():
    r = run_baseline()
    assert r["rows"] == 592
    assert r["failures"] == 48


def test_baseline_never_predicts_a_failure():
    r = run_baseline()
    assert r["predicted_failures"] == 0
    assert r["recall"] == 0.0


def test_baseline_accuracy_looks_good_but_is_useless():
    r = run_baseline()
    assert r["accuracy"] == pytest.approx(136 / 148)


def test_test_split_contains_expected_failures():
    assert run_baseline()["test_failures"] == 12
