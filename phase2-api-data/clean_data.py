from pathlib import Path

import pandas as pd

from common import is_blank, silent_failure_mask


# ==================================================
# 1. Configuration
# ==================================================

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "chat_logs.csv"
CLEANED_PATH = DATA_DIR / "chat_logs_cleaned.csv"

REQUIRED_COLUMNS = [
    "id",
    "timestamp",
    "model",
    "category",
    "prompt",
    "response",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "error",
    "judge_score",
]

cleaning_log = []


# ==================================================
# 2. Cleaning log
# ==================================================

def log_step(
    step_name,
    rows_before,
    rows_after,
    reason,
    values_changed=0,
):
    cleaning_log.append(
        {
            "step": step_name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "values_changed": values_changed,
            "reason": reason,
        }
    )


# ==================================================
# 3. Cleaning functions
# ==================================================

def normalize_text_columns(df):
    """Normalize model and category before deduplication."""
    df = df.copy()
    rows_before = len(df)

    model_before = df["model"].copy()
    category_before = df["category"].copy()

    df["model"] = df["model"].str.strip().str.lower()
    df["category"] = df["category"].str.strip().str.lower()

    model_changes = int(
        (model_before.fillna("") != df["model"].fillna("")).sum()
    )
    category_changes = int(
        (category_before.fillna("") != df["category"].fillna("")).sum()
    )

    values_changed = model_changes + category_changes

    log_step(
        step_name="Normalize model and category",
        rows_before=rows_before,
        rows_after=len(df),
        reason=(
            f"Standardized text. Model values changed: {model_changes}; "
            f"category values changed: {category_changes}."
        ),
        values_changed=values_changed,
    )

    return df


def remove_exact_duplicates(df):
    """Remove exact duplicate records after normalization."""
    rows_before = len(df)
    df = df.drop_duplicates().copy()

    log_step(
        step_name="Remove exact duplicate rows",
        rows_before=rows_before,
        rows_after=len(df),
        reason="Remove identical records after normalization.",
    )

    return df


def remove_invalid_latency(df):
    """Remove rows with missing or negative latency."""
    rows_before = len(df)

    invalid_mask = df["latency_ms"].isna() | df["latency_ms"].lt(0)
    invalid_count = int(invalid_mask.sum())

    df = df.loc[~invalid_mask].copy()

    log_step(
        step_name="Remove invalid latency rows",
        rows_before=rows_before,
        rows_after=len(df),
        reason=(
            f"Removed {invalid_count} rows with missing or negative latency."
        ),
    )

    return df


def flag_silent_failures(df):
    """
    Flag records with no response and no error.
    Keep these records for further investigation.
    """
    df = df.copy()
    rows_before = len(df)

    df["is_silent_failure"] = silent_failure_mask(df)
    silent_count = int(df["is_silent_failure"].sum())

    log_step(
        step_name="Flag silent failures",
        rows_before=rows_before,
        rows_after=len(df),
        reason=(
            f"Flagged {silent_count} records without response or error; "
            "records were retained."
        ),
        values_changed=silent_count,
    )

    return df


# ==================================================
# 4. Data validation
# ==================================================

def validate_data(df):
    """Raise ValueError when a required invariant is violated."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if "is_silent_failure" not in df.columns:
        raise ValueError("Missing is_silent_failure column.")

    if df.duplicated().any():
        raise ValueError("Exact duplicate rows remain.")

    if df["id"].isna().any():
        raise ValueError("Missing IDs found.")

    if not df["id"].is_unique:
        raise ValueError("IDs must be unique.")

    if df["timestamp"].isna().any():
        raise ValueError("Missing or invalid timestamps found.")

    for column in ("model", "category"):
        if df[column].isna().any():
            raise ValueError(f"Missing {column} values found.")

        empty_mask = (
            df[column].astype("string").str.strip().eq("").fillna(False)
        )
        if empty_mask.any():
            raise ValueError(f"Empty {column} values found.")

    if df["latency_ms"].isna().any():
        raise ValueError("Missing latency values found.")

    if df["latency_ms"].lt(0).any():
        raise ValueError("Negative latency values found.")

    if df["is_silent_failure"].isna().any():
        raise ValueError("Missing silent-failure flags found.")

    for column in ("input_tokens", "output_tokens"):
        if df[column].dropna().lt(0).any():
            raise ValueError(f"Negative {column} counts found.")

    valid_scores = df["judge_score"].dropna()
    if not valid_scores.between(1, 5).all():
        raise ValueError("Judge scores outside the expected range 1–5.")


# ==================================================
# 5. Audit helpers
# ==================================================

def audit_response_and_error(df):
    """Audit response/error consistency and silent failures."""

    response_empty = is_blank(df["response"])
    error_empty = is_blank(df["error"])

    print("\nResponse/error consistency audit:")
    print(
        "Response present, no error:",
        int((~response_empty & error_empty).sum()),
    )
    print(
        "Response missing, error present:",
        int((response_empty & ~error_empty).sum()),
    )
    print(
        "Response missing, no error:",
        int((response_empty & error_empty).sum()),
    )
    print(
        "Response present, error present:",
        int((~response_empty & ~error_empty).sum()),
    )

    print("\nRecords with missing response and error:")
    print(
        df.loc[
            response_empty & error_empty,
            [
                "id",
                "timestamp",
                "model",
                "category",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "judge_score",
                "is_silent_failure",
            ],
        ].to_string(index=False)
    )

    missing_response_with_score = response_empty & df["judge_score"].notna()

    print("\nMissing response / judge score audit:")
    print(
        "Rows with missing response but present judge score:",
        int(missing_response_with_score.sum()),
    )

    failed_call_mask = response_empty & ~error_empty
    successful_call_mask = ~response_empty & error_empty

    print("\nMissing values in failed API calls:")
    print(
        "Missing input_tokens:",
        int((failed_call_mask & df["input_tokens"].isna()).sum()),
    )
    print(
        "Missing output_tokens:",
        int((failed_call_mask & df["output_tokens"].isna()).sum()),
    )
    print(
        "Missing judge_score:",
        int((failed_call_mask & df["judge_score"].isna()).sum()),
    )

    print("\nMissing values in calls with a response and no error:")
    print(
        "Missing input_tokens:",
        int((successful_call_mask & df["input_tokens"].isna()).sum()),
    )
    print(
        "Missing output_tokens:",
        int((successful_call_mask & df["output_tokens"].isna()).sum()),
    )
    print(
        "Missing judge_score:",
        int((successful_call_mask & df["judge_score"].isna()).sum()),
    )

    print("\nError type distribution:")
    print(df["error"].value_counts(dropna=False).to_string())


def audit_dataset(df):
    """Run exploratory data quality checks."""

    print("\nHigh latency audit:")
    print("Maximum latency (ms):", df["latency_ms"].max())
    print("Rows above 30,000 ms:", int(df["latency_ms"].gt(30_000).sum()))
    print("Rows above 60,000 ms:", int(df["latency_ms"].gt(60_000).sum()))
    print("Rows above 100,000 ms:", int(df["latency_ms"].gt(100_000).sum()))

    print("\nFive slowest records:")
    print(
        df.nlargest(5, "latency_ms")[
            ["id", "model", "category", "latency_ms", "error"]
        ].to_string(index=False)
    )

    duplicate_id_mask = df["id"].duplicated(keep=False)

    print("\nDuplicate ID audit:")
    print("Rows with a duplicated ID:", int(duplicate_id_mask.sum()))
    print(
        "Distinct duplicated IDs:",
        int(df.loc[duplicate_id_mask, "id"].nunique()),
    )

    if duplicate_id_mask.any():
        print(
            df.loc[
                duplicate_id_mask,
                ["id", "timestamp", "model", "category", "latency_ms", "error"],
            ]
            .sort_values("id")
            .to_string(index=False)
        )

    print("\nMissing values audit:")
    print(df.isna().sum().to_string())

    print("\nMissing values percentage:")
    print((df.isna().mean() * 100).round(2).to_string())

    print("\nToken count and judge score audit:")
    print("Negative input_tokens:", int(df["input_tokens"].lt(0).sum()))
    print("Negative output_tokens:", int(df["output_tokens"].lt(0).sum()))
    print("Minimum judge_score:", df["judge_score"].min())
    print("Maximum judge_score:", df["judge_score"].max())

    print("\nData types audit:")
    print(df.dtypes.to_string())

    print("\nTimestamp range audit:")
    print("Earliest timestamp:", df["timestamp"].min())
    print("Latest timestamp:", df["timestamp"].max())
    print("Missing timestamps:", int(df["timestamp"].isna().sum()))

    print("\nLatency validity audit:")
    print("Zero latency values:", int(df["latency_ms"].eq(0).sum()))
    print("Missing latency values:", int(df["latency_ms"].isna().sum()))

    print("\nID validity audit:")
    print("Missing IDs:", int(df["id"].isna().sum()))
    print("Duplicate IDs:", int(df["id"].duplicated().sum()))

    print("\nModel and category validity audit:")
    print("Missing model values:", int(df["model"].isna().sum()))
    print(
        "Empty model values:",
        int(df["model"].astype("string").str.strip().eq("").sum()),
    )
    print("Missing category values:", int(df["category"].isna().sum()))
    print(
        "Empty category values:",
        int(df["category"].astype("string").str.strip().eq("").sum()),
    )

    print("\nModels:")
    print(df["model"].value_counts(dropna=False).to_string())

    print("\nCategories:")
    print(df["category"].value_counts(dropna=False).to_string())


# ==================================================
# 6. Main pipeline
# ==================================================

def main():
    global cleaning_log
    cleaning_log = []

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    # Load original data.
    df = pd.read_csv(CSV_PATH)

    print("Original row count:", len(df))
    print("Original column count:", len(df.columns))
    print("Original columns:", df.columns.tolist())

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing_columns)}"
        )

    log_step(
        step_name="Initial dataset",
        rows_before=len(df),
        rows_after=len(df),
        reason="Loaded original CSV; no cleaning applied.",
    )

    # Convert data types before cleaning and auditing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for column in (
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "judge_score",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print("\nTimestamp format audit:")
    print(df["timestamp"].head(5).to_string(index=False))
    print("\nTimestamp type after conversion:")
    print(df["timestamp"].dtype)

    # Normalize before deduplication.
    df = normalize_text_columns(df)

    print("\nModels after normalization:")
    print(df["model"].value_counts(dropna=False).to_string())
    print("\nCategories after normalization:")
    print(df["category"].value_counts(dropna=False).to_string())

    # Remove exact duplicates after normalization.
    df = remove_exact_duplicates(df)

    print("\nAfter removing exact duplicates:")
    print("Rows remaining:", len(df))

    print("\nNegative latency check after deduplication:")
    print("Negative latency rows:", int(df["latency_ms"].lt(0).sum()))
    print(
        df.loc[df["latency_ms"].lt(0), ["id", "latency_ms"]].to_string(
            index=False
        )
    )

    # Audit high latency before removing invalid measurements.
    audit_dataset(df)

    # Remove missing and negative latency values.
    df = remove_invalid_latency(df)

    print("\nAfter removing invalid latency values:")
    print("Rows remaining:", len(df))

    # Flag, rather than discard, silent failures.
    df = flag_silent_failures(df)

    print("\nSilent failure audit:")
    print("Silent failures:", int(df["is_silent_failure"].sum()))
    print("Silent failure IDs:", df.loc[df["is_silent_failure"], "id"].to_list())

    audit_response_and_error(df)

    # Validate before saving.
    validate_data(df)
    print("\nData validation passed (in-memory dataset).")

    # Save the cleaned dataset without modifying the original.
    df.to_csv(CLEANED_PATH, index=False)

    print("\nCleaned dataset saved to:", CLEANED_PATH)
    print("Final row count:", len(df))
    print("Final column count:", len(df.columns))

    # Reload the saved CSV and validate it independently.
    df_check = pd.read_csv(CLEANED_PATH, parse_dates=["timestamp"])

    for column in (
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "judge_score",
    ):
        df_check[column] = pd.to_numeric(df_check[column], errors="coerce")

    df_check["is_silent_failure"] = (
        df_check["is_silent_failure"]
        .astype("string")
        .str.lower()
        .map({"true": True, "false": False})
    )

    validate_data(df_check)
    print("Data validation passed (reloaded CSV).")

    if len(df_check) != len(df):
        raise ValueError(
            "Saved CSV row count differs from the in-memory dataset."
        )

    if len(df_check.columns) != len(df.columns):
        raise ValueError(
            "Saved CSV column count differs from the in-memory dataset."
        )

    if df_check["id"].astype(str).tolist() != df["id"].astype(str).tolist():
        raise ValueError(
            "Saved CSV IDs or row order differ from the in-memory dataset."
        )

    if (
        df_check["is_silent_failure"].tolist()
        != df["is_silent_failure"].tolist()
    ):
        raise ValueError("Silent-failure flags changed after saving.")

    print("\nSaved dataset validation:")
    print("Rows:", len(df_check))
    print("Columns:", len(df_check.columns))
    print("Duplicate IDs:", int(df_check["id"].duplicated().sum()))
    print("Negative latency values:", int(df_check["latency_ms"].lt(0).sum()))
    print("Missing timestamps:", int(df_check["timestamp"].isna().sum()))
    print("Silent failures:", int(df_check["is_silent_failure"].sum()))

    print("\nCleaning log:")
    print(pd.DataFrame(cleaning_log).to_string(index=False))

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()
