import pandas as pd
import pytest

import report


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def sample_data():
    """
    Four requests. Derived columns are built by the real report code,
    so these tests exercise the same rules as production.

    id 3 failed with an error (no score). id 4 is a silent failure that
    still carries a judge_score of 5: the score must never be counted.
    """
    raw = pd.DataFrame([
        {"id": 1, "model": "gpt-5", "category": "factual",
         "latency_ms": 100, "response": "OK", "error": None,
         "judge_score": 4.0},
        {"id": 2, "model": "gpt-5-mini", "category": "factual",
         "latency_ms": 200, "response": "OK", "error": None,
         "judge_score": 2.0},
        {"id": 3, "model": "gpt-5", "category": "adversarial",
         "latency_ms": 300, "response": None, "error": "TimeoutError",
         "judge_score": None},
        {"id": 4, "model": "gpt-5-mini", "category": "adversarial",
         "latency_ms": 400, "response": None, "error": None,
         "judge_score": 5.0},
    ])
    return report.prepare_columns(raw)


@pytest.fixture
def cleaned_csv(monkeypatch, tmp_path):
    """Point the report at a small CSV, so tests never need real output files."""
    path = tmp_path / "chat_logs_cleaned.csv"
    pd.DataFrame([
        {"model": "gpt-5", "category": "factual", "response": "OK",
         "error": None, "latency_ms": 120, "judge_score": 4.0},
        {"model": "gpt-5", "category": "factual", "response": None,
         "error": None, "latency_ms": 90, "judge_score": 5.0},
        {"model": "gpt-5", "category": "factual", "response": "  ",
         "error": "  ", "latency_ms": 80, "judge_score": 5.0},
        {"model": "gpt-5", "category": "factual", "response": None,
         "error": "TimeoutError", "latency_ms": 30000, "judge_score": None},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(report, "INPUT_PATH", path)
    return path


def scored_frame(group_scores, rest_scores):
    """Build a frame of successful, scored calls for the comparison tests."""
    rows = [
        {"category": "adversarial", "latency_ms": 1, "response": "OK",
         "error": None, "judge_score": score}
        for score in group_scores
    ] + [
        {"category": "factual", "latency_ms": 1, "response": "OK",
         "error": None, "judge_score": score}
        for score in rest_scores
    ]
    return report.prepare_columns(pd.DataFrame(rows))


# ==================================================
# prepare_columns
# ==================================================

def test_prepare_columns_classifies_every_row_once(sample_data):
    assert sample_data["is_success"].tolist() == [True, True, False, False]
    assert sample_data["has_error"].tolist() == [False, False, True, False]
    assert sample_data["is_silent_failure"].tolist() == [False, False, False, True]


def test_prepare_columns_does_not_modify_the_input():
    raw = pd.DataFrame([
        {"model": "m", "category": "c", "latency_ms": 1,
         "response": "OK", "error": None},
    ])

    report.prepare_columns(raw)

    assert "has_error" not in raw.columns


# ==================================================
# calculate_metrics
# ==================================================

def test_overall_metrics(sample_data):
    result = report.calculate_metrics(sample_data)

    assert result.loc[0, "total_requests"] == 4
    assert result.loc[0, "mean_latency_ms"] == pytest.approx(250)
    assert result.loc[0, "max_latency_ms"] == 400
    assert result.loc[0, "error_count"] == 1
    assert result.loc[0, "error_rate_pct"] == pytest.approx(25)
    assert result.loc[0, "silent_failure_count"] == 1
    assert result.loc[0, "silent_failure_rate_pct"] == pytest.approx(25)


def test_metrics_by_model(sample_data):
    by_model = report.calculate_metrics(sample_data, "model").set_index("model")

    assert by_model.loc["gpt-5", "total_requests"] == 2
    assert by_model.loc["gpt-5", "mean_latency_ms"] == pytest.approx(200)
    assert by_model.loc["gpt-5-mini", "total_requests"] == 2
    assert by_model.loc["gpt-5-mini", "mean_latency_ms"] == pytest.approx(300)


def test_metrics_by_category(sample_data):
    by_category = report.calculate_metrics(sample_data, "category").set_index("category")

    assert by_category.loc["factual", "total_requests"] == 2
    assert by_category.loc["adversarial", "total_requests"] == 2
    assert by_category.loc["adversarial", "error_count"] == 1
    assert by_category.loc["adversarial", "silent_failure_count"] == 1


def test_percentile_metrics(sample_data):
    result = report.calculate_metrics(sample_data)

    assert result.loc[0, "p50_latency_ms"] == pytest.approx(250)
    assert result.loc[0, "p95_latency_ms"] == pytest.approx(385)


def test_success_latency_excludes_failed_calls(sample_data):
    """Only ids 1 and 2 succeeded: latencies 100 and 200, p95 = 195."""
    result = report.calculate_metrics(sample_data)

    assert result.loc[0, "success_count"] == 2
    assert result.loc[0, "p95_latency_success_ms"] == pytest.approx(195)
    assert result.loc[0, "p95_latency_success_ms"] < result.loc[0, "p95_latency_ms"]


def test_zero_errors_has_zero_error_rate():
    raw = pd.DataFrame([
        {"model": "m", "category": "c", "latency_ms": 10,
         "response": "OK", "error": None, "judge_score": 3.0},
        {"model": "m", "category": "c", "latency_ms": 20,
         "response": "OK", "error": None, "judge_score": 4.0},
    ])

    result = report.calculate_metrics(report.prepare_columns(raw))

    assert result.loc[0, "error_count"] == 0
    assert result.loc[0, "error_rate_pct"] == pytest.approx(0)
    assert result.loc[0, "silent_failure_count"] == 0
    assert result.loc[0, "silent_failure_rate_pct"] == pytest.approx(0)


# ==================================================
# judge_score metrics
# ==================================================

def test_overall_mean_judge_score_uses_successful_calls_only(sample_data):
    """ids 1 and 2 count (4 and 2). id 4's score of 5 must be ignored."""
    result = report.calculate_metrics(sample_data)

    assert result.loc[0, "scored_count"] == 2
    assert result.loc[0, "mean_judge_score"] == pytest.approx(3.0)


def test_mean_judge_score_by_model(sample_data):
    by_model = report.calculate_metrics(sample_data, "model").set_index("model")

    assert by_model.loc["gpt-5", "mean_judge_score"] == pytest.approx(4.0)
    assert by_model.loc["gpt-5", "scored_count"] == 1
    # gpt-5-mini has a silent failure scored 5; only the real answer (2) counts.
    assert by_model.loc["gpt-5-mini", "mean_judge_score"] == pytest.approx(2.0)
    assert by_model.loc["gpt-5-mini", "scored_count"] == 1


def test_mean_judge_score_by_category_with_no_scored_rows(sample_data):
    by_category = report.calculate_metrics(sample_data, "category").set_index("category")

    assert by_category.loc["factual", "mean_judge_score"] == pytest.approx(3.0)
    assert by_category.loc["adversarial", "scored_count"] == 0
    assert pd.isna(by_category.loc["adversarial", "mean_judge_score"])


# ==================================================
# compare_group_scores
# ==================================================

def test_comparison_detects_a_clearly_lower_group():
    df = scored_frame([1, 2] * 20, [4, 5] * 20)

    result = report.compare_group_scores(df, "category", "adversarial")

    assert result["n_group"] == 40
    assert result["n_rest"] == 40
    assert result["mean_group"] == pytest.approx(1.5)
    assert result["mean_rest"] == pytest.approx(4.5)
    assert result["difference"] == pytest.approx(-3.0)
    assert result["ci95_high"] < 0
    assert result["p_value_lower"] < 0.01


def test_comparison_sees_no_difference_between_identical_groups():
    df = scored_frame([3, 4] * 20, [3, 4] * 20)

    result = report.compare_group_scores(df, "category", "adversarial")

    assert result["difference"] == pytest.approx(0.0)
    assert result["ci95_low"] < 0 < result["ci95_high"]
    assert result["p_value_lower"] > 0.05


def test_comparison_ignores_unscored_and_failed_calls():
    df = scored_frame([1, 1, 1], [5, 5, 5])
    extra = report.prepare_columns(pd.DataFrame([
        # silent failure carrying a top score, in the adversarial group
        {"category": "adversarial", "latency_ms": 1, "response": None,
         "error": None, "judge_score": 5.0},
        # failed call, no score
        {"category": "adversarial", "latency_ms": 1, "response": None,
         "error": "TimeoutError", "judge_score": None},
    ]))
    df = pd.concat([df, extra], ignore_index=True)

    result = report.compare_group_scores(df, "category", "adversarial")

    assert result["n_group"] == 3
    assert result["mean_group"] == pytest.approx(1.0)


def test_comparison_interval_has_the_expected_95_percent_width():
    """
    Two groups of 40 scores spread evenly over 1-5 (variance 2 each).
    Standard error of the difference = sqrt(2/40 + 2/40) = 0.316,
    so a 95% interval is about 2 * 1.96 * 0.316 = 1.24 wide.
    A 50% interval would be only about 0.43 wide.
    """
    df = scored_frame([1, 2, 3, 4, 5] * 8, [1, 2, 3, 4, 5] * 8)

    result = report.compare_group_scores(df, "category", "adversarial")

    width = result["ci95_high"] - result["ci95_low"]
    assert width == pytest.approx(1.24, abs=0.15)


def test_comparison_is_reproducible_with_the_same_seed():
    df = scored_frame([1, 2, 3, 4] * 5, [3, 4, 5] * 7)

    first = report.compare_group_scores(df, "category", "adversarial", seed=1)
    second = report.compare_group_scores(df, "category", "adversarial", seed=1)

    assert first == second


def test_comparison_needs_enough_scored_rows():
    df = scored_frame([1], [4, 5, 4, 5])

    with pytest.raises(ValueError, match="Not enough scored rows"):
        report.compare_group_scores(df, "category", "adversarial")


@pytest.mark.parametrize("group, rest, expected", [
    ([1, 2] * 20, [4, 5] * 20, "score lower"),
    ([4, 5] * 20, [1, 2] * 20, "score higher"),
    ([3, 4] * 20, [3, 4] * 20, "cannot say"),
])
def test_describe_comparison_matches_the_numbers(group, rest, expected):
    df = scored_frame(group, rest)
    result = report.compare_group_scores(df, "category", "adversarial")

    text = report.describe_comparison(result)

    assert expected in text
    assert "adversarial" in text


# ==================================================
# load_data
# ==================================================

def test_load_data_creates_derived_columns(cleaned_csv):
    df = report.load_data()

    assert len(df) == 4
    for column in ("has_error", "is_silent_failure", "is_success"):
        assert pd.api.types.is_bool_dtype(df[column].dtype)


def test_load_data_recomputes_silent_failures_from_response_and_error(cleaned_csv):
    """Blank strings and missing values both count; a logged error does not."""
    df = report.load_data()

    assert df["is_silent_failure"].tolist() == [False, True, True, False]
    assert df["has_error"].tolist() == [False, False, False, True]


def test_load_data_raises_error_when_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "INPUT_PATH", tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="Cleaned dataset not found"):
        report.load_data()


def test_load_data_raises_error_when_required_columns_are_missing(
    monkeypatch, tmp_path
):
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"model": ["gpt-5"], "category": ["factual"]}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(report, "INPUT_PATH", csv_path)

    with pytest.raises(ValueError, match="Missing required columns"):
        report.load_data()


def test_load_data_requires_the_judge_score_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "no_scores.csv"
    pd.DataFrame({
        "model": ["gpt-5"],
        "category": ["factual"],
        "response": ["OK"],
        "error": [None],
        "latency_ms": [10],
    }).to_csv(csv_path, index=False)
    monkeypatch.setattr(report, "INPUT_PATH", csv_path)

    with pytest.raises(ValueError, match="judge_score"):
        report.load_data()


def test_load_data_raises_error_for_invalid_latency(monkeypatch, tmp_path):
    csv_path = tmp_path / "invalid_latency.csv"
    pd.DataFrame({
        "model": ["gpt-5"],
        "category": ["factual"],
        "response": ["OK"],
        "error": [None],
        "latency_ms": ["not-a-number"],
        "judge_score": [4.0],
    }).to_csv(csv_path, index=False)
    monkeypatch.setattr(report, "INPUT_PATH", csv_path)

    with pytest.raises(ValueError, match="not-a-number"):
        report.load_data()
