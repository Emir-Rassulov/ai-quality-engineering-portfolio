import pandas as pd
import pytest

import ml_data
from ml_data import load_data

COLUMNS = [
    "id", "model", "category", "prompt", "response", "error",
    "input_tokens", "output_tokens", "latency_ms", "judge_score",
]


def make_rows():
    """Three rows: success, logged error, silent failure."""
    return [
        [1, "gpt-5", "factual", "Hello QA", "Fine answer", "", 10, 20, 500.0, 4],
        [2, "gpt-5", "factual", "Second", "", "timeout", 10, 0, 900.0, None],
        [3, "gpt-5-mini", "creative", "Third", "", "", 12, 0, 800.0, None],
    ]


@pytest.fixture
def use_csv(tmp_path, monkeypatch):
    """Write a custom CSV and point ml_data at it."""
    def _use(rows=None, drop=None):
        df = pd.DataFrame(make_rows() if rows is None else rows, columns=COLUMNS)
        if drop:
            df = df.drop(columns=[drop])
        path = tmp_path / "logs.csv"
        df.to_csv(path, index=False)
        monkeypatch.setattr(ml_data, "DATA_PATH", path)
    return _use


def test_target_for_success_error_and_silent_failure(use_csv):
    use_csv()
    result = load_data()
    assert result["real_failure"].tolist() == [0, 1, 1]


def test_prompt_length_is_calculated(use_csv):
    use_csv()
    assert load_data().loc[0, "prompt_length"] == 8


def test_prompt_length_is_zero_when_prompt_is_missing(use_csv):
    rows = make_rows()
    rows[0][3] = None
    use_csv(rows)
    assert load_data().loc[0, "prompt_length"] == 0


def test_response_length_is_zero_for_blank_response(use_csv):
    use_csv()
    assert load_data()["response_length"].tolist() == [11, 0, 0]


@pytest.mark.parametrize(
    "missing_column", ["model", "category", "prompt", "response", "error", "input_tokens"]
)
def test_missing_required_column_raises(use_csv, missing_column):
    use_csv(drop=missing_column)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_data()


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_data, "DATA_PATH", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_data()


def test_empty_dataset_raises(use_csv):
    use_csv(rows=[])
    with pytest.raises(ValueError, match="empty"):
        load_data()


def test_single_class_target_raises(use_csv):
    rows = [[r[0], r[1], r[2], r[3], "Fine answer", "", *r[6:]] for r in make_rows()]
    use_csv(rows)
    with pytest.raises(ValueError, match="both classes"):
        load_data()
