"""Shared rules used by both clean_data.py and report.py."""

import pandas as pd


def is_blank(series: pd.Series) -> pd.Series:
    """True where a value is missing, empty, or whitespace only."""
    blank = series.isna() | series.astype("string").str.strip().eq("").fillna(False)
    return blank.astype(bool)


def silent_failure_mask(df: pd.DataFrame) -> pd.Series:
    """A silent failure has no response AND no error: nothing explains it."""
    return is_blank(df["response"]) & is_blank(df["error"])
