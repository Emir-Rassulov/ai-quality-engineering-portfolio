"""Single source of truth for the data, the target, the features and the split."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "phase2-api-data"))
from common import is_blank, silent_failure_mask  # noqa: E402

DATA_PATH = PROJECT_ROOT / "phase2-api-data" / "chat_logs_cleaned.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "real_failure"

REQUIRED_COLUMNS = {"model", "category", "prompt", "response", "error", "input_tokens"}

# Features known BEFORE the call is made. This is the only list the model may use.
CATEGORICAL_FEATURES = ["model", "category"]
NUMERIC_FEATURES = ["prompt_length", "input_tokens"]
PRE_CALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Features that only exist AFTER the call (or are the label itself).
FORBIDDEN_FEATURES = {
    "response",
    "error",
    "output_tokens",
    "latency_ms",
    "judge_score",
    "response_length",
}


def require_columns(df: pd.DataFrame, columns) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_data() -> pd.DataFrame:
    """Read the cleaned logs and add the target and the length features."""
    df = pd.read_csv(DATA_PATH)

    require_columns(df, REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("Dataset is empty")

    df[TARGET] = (~is_blank(df["error"]) | silent_failure_mask(df)).astype(int)

    if df[TARGET].nunique() < 2:
        raise ValueError("Target needs both classes (failure and success)")

    df["prompt_length"] = df["prompt"].fillna("").str.len()
    df["response_length"] = df["response"].fillna("").str.len()
    return df


def split_data(df: pd.DataFrame, features=None):
    """Seeded, stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    features = PRE_CALL_FEATURES if features is None else features
    return train_test_split(
        df[features],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
