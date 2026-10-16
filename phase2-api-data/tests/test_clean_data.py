import pandas as pd
import pytest

import clean_data


# ==================================================
# Fixtures and helpers
# ==================================================

@pytest.fixture(autouse=True)
def clear_cleaning_log():
    """Keep cleaning log state isolated between tests."""
    clean_data.cleaning_log.clear()
    yield
    clean_data.cleaning_log.clear()


def make_row(**overrides):
    """Create a valid row with optional field overrides."""
    row = {
        "id": 1,
        "timestamp": pd.Timestamp("2026-10-01 12:00:00"),
        "model": "gpt-5",
        "category": "factual",
        "prompt": "Test prompt",
        "response": "Test response",
        "latency_ms": 100.0,
        "input_tokens": 10,
        "output_tokens": 20,
        "error": None,
        "judge_score": 4.0,
    }
    row.update(overrides)
    return row


def make_valid_frame(*rows):
    """Build a frame that passes validate_data, then add the flag column."""
    df = pd.DataFrame(rows or [make_row()])
    return clean_data.flag_silent_failures(df)


# ==================================================
# Normalization and deduplication
# ==================================================

def test_normalization_then_deduplication():
    """Normalize text before removing exact duplicate rows."""
    df = pd.DataFrame([
        make_row(model=" GPT-5 ", category=" Factual "),
        make_row(model="gpt-5", category="factual"),
    ])

    normalized = clean_data.normalize_text_columns(df)
    result = clean_data.remove_exact_duplicates(normalized)

    assert result["model"].iloc[0] == "gpt-5"
    assert result["category"].iloc[0] == "factual"
    assert len(result) == 1


def test_normalization_is_logged():
    """Record how many text values were normalized."""
    df = pd.DataFrame([make_row(model=" GPT-5 ", category=" Factual ")])

    clean_data.normalize_text_columns(df)

    assert len(clean_data.cleaning_log) == 1
    assert clean_data.cleaning_log[0]["values_changed"] == 2


# ==================================================
# Latency cleaning
# ==================================================

def test_removes_negative_latency():
    df = pd.DataFrame([
        make_row(id=1, latency_ms=-1),
        make_row(id=2, latency_ms=100),
    ])

    result = clean_data.remove_invalid_latency(df)

    assert result["id"].tolist() == [2]


def test_removes_nan_latency():
    df = pd.DataFrame([
        make_row(id=1, latency_ms=float("nan")),
        make_row(id=2, latency_ms=100),
    ])

    result = clean_data.remove_invalid_latency(df)

    assert result["id"].tolist() == [2]
    assert result["latency_ms"].notna().all()


# ==================================================
# Silent failure detection
# ==================================================

def test_flags_silent_failures_without_dropping_rows():
    df = pd.DataFrame([
        make_row(id=1, response=None, error=None),
        make_row(id=2, response="Valid response", error=None),
        make_row(id=3, response=None, error="TimeoutError"),
    ])

    result = clean_data.flag_silent_failures(df)

    assert len(result) == 3
    assert result["is_silent_failure"].tolist() == [True, False, False]


@pytest.mark.parametrize("response, error, expected", [
    ("   ", None, True),        # whitespace-only response, no error
    ("", None, True),           # empty response, no error
    (None, "  ", True),         # blank error counts as no error
    ("   ", "TimeoutError", False),   # an error explains the missing response
])
def test_silent_failure_treats_blank_strings_as_missing(response, error, expected):
    df = pd.DataFrame([make_row(response=response, error=error)])

    result = clean_data.flag_silent_failures(df)

    assert bool(result["is_silent_failure"].iloc[0]) is expected


# ==================================================
# Data validation
# ==================================================

def test_validate_data_accepts_a_valid_frame():
    """Guards the failure tests below: a good frame must pass."""
    clean_data.validate_data(make_valid_frame())


def test_validate_data_rejects_duplicate_ids():
    """Reject duplicate IDs even when the rows differ."""
    df = make_valid_frame(
        make_row(id=1, prompt="First prompt"),
        make_row(id=1, prompt="Different prompt"),
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        clean_data.validate_data(df)


def test_validate_data_rejects_exact_duplicate_rows():
    """Identical rows are reported as exact duplicates, not as an ID problem."""
    df = make_valid_frame(make_row(id=1), make_row(id=1))

    with pytest.raises(ValueError, match="Exact duplicate rows"):
        clean_data.validate_data(df)


def test_validate_data_rejects_missing_flag_column():
    df = pd.DataFrame([make_row()])

    with pytest.raises(ValueError, match="Missing is_silent_failure column"):
        clean_data.validate_data(df)


@pytest.mark.parametrize("override, message", [
    ({"latency_ms": -5.0}, "Negative latency"),
    ({"latency_ms": float("nan")}, "Missing latency"),
    ({"judge_score": 9.0}, "outside the expected range"),
    ({"judge_score": 0.0}, "outside the expected range"),
    ({"input_tokens": -1}, "Negative input_tokens"),
    ({"output_tokens": -1}, "Negative output_tokens"),
    ({"model": "   "}, "Empty model"),
    ({"category": None}, "Missing category"),
    ({"timestamp": pd.NaT}, "invalid timestamps"),
])
def test_validate_data_rejects_bad_values(override, message):
    df = make_valid_frame(make_row(**override))

    with pytest.raises(ValueError, match=message):
        clean_data.validate_data(df)
