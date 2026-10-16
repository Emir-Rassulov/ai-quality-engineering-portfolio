from pathlib import Path

import numpy as np
import pandas as pd

from common import is_blank, silent_failure_mask


# ==================================================
# 1. Configuration
# ==================================================

DATA_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATA_DIR / "chat_logs_cleaned.csv"

MODEL_REPORT_PATH = DATA_DIR / "report_by_model.csv"
CATEGORY_REPORT_PATH = DATA_DIR / "report_by_category.csv"
OVERALL_REPORT_PATH = DATA_DIR / "report_overall.csv"
ADVERSARIAL_REPORT_PATH = DATA_DIR / "report_adversarial_vs_rest.csv"

REQUIRED_COLUMNS = [
    "model",
    "category",
    "response",
    "error",
    "latency_ms",
    "judge_score",
]


# ==================================================
# 2. Load and prepare data
# ==================================================

def prepare_columns(df):
    """
    Derive the analysis columns from response and error.

    - has_error:         the API call logged an error
    - is_silent_failure: no response and no error (nothing explains it)
    - is_success:        neither of the above

    The rule lives in common.py, so cleaning and reporting always agree.
    """
    df = df.copy()
    df["has_error"] = ~is_blank(df["error"])
    df["is_silent_failure"] = silent_failure_mask(df)
    df["is_success"] = ~df["has_error"] & ~df["is_silent_failure"]
    return df


def load_data():
    """Load the cleaned dataset and prepare analysis columns."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}. "
            "Run clean_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="raise")

    return prepare_columns(df)


# ==================================================
# 3. Calculate metrics
# ==================================================

def calculate_metrics(df, group_column=None):
    """
    Calculate latency percentiles, error rate, and silent-failure rate.

    Latency is reported for all calls and for successful calls only.
    Failed calls mostly measure the timeout setting, not model speed,
    so the success-only p95 is the number to compare models with.

    mean_judge_score uses successful calls only. A silent failure has no
    response, so any score it carries cannot be a score of an answer.
    scored_count says how many rows the mean is based on.

    If group_column is provided, calculate metrics for each group.
    """

    if group_column is None:
        groups = [("overall", df)]
    else:
        groups = df.groupby(group_column, dropna=False)

    results = []

    for group_name, group in groups:
        total_requests = len(group)

        if total_requests == 0:
            continue

        error_count = int(group["has_error"].sum())
        silent_failure_count = int(group["is_silent_failure"].sum())
        successful = group.loc[group["is_success"], "latency_ms"]
        scores = group.loc[group["is_success"], "judge_score"].dropna()

        results.append(
            {
                group_column or "scope": group_name,
                "total_requests": total_requests,
                "mean_latency_ms": round(group["latency_ms"].mean(), 2),
                "p50_latency_ms": round(group["latency_ms"].quantile(0.50), 2),
                "p95_latency_ms": round(group["latency_ms"].quantile(0.95), 2),
                "max_latency_ms": round(group["latency_ms"].max(), 2),
                "success_count": len(successful),
                "p95_latency_success_ms": round(successful.quantile(0.95), 2),
                "scored_count": len(scores),
                "mean_judge_score": round(scores.mean(), 2),
                "error_count": error_count,
                "error_rate_pct": round(error_count / total_requests * 100, 2),
                "silent_failure_count": silent_failure_count,
                "silent_failure_rate_pct": round(
                    silent_failure_count / total_requests * 100, 2
                ),
            }
        )

    return pd.DataFrame(results)


# ==================================================
# 4. Compare one group with the rest
# ==================================================

def compare_group_scores(
    df,
    group_column,
    group_value,
    n_resamples=10_000,
    seed=42,
):
    """
    Does one group (for example adversarial prompts) score lower than the rest?

    Only scored, successful calls count. Returns the sample sizes, both means,
    their difference, a 95% bootstrap interval for the difference, and a
    one-sided permutation p-value for "the group scores lower".

    Judge scores are 1-5 ratings, so no normal-distribution assumption is made:
    both numbers come from resampling the data itself. A fixed seed keeps the
    result reproducible. The smallest p-value this method can report is
    1 / (n_resamples + 1), so read 0.0001 as "below 0.0001".
    """
    scored = df[df["is_success"] & df["judge_score"].notna()]
    in_group = (scored[group_column] == group_value).to_numpy()
    values = scored["judge_score"].to_numpy(dtype=float)

    group_scores = values[in_group]
    rest_scores = values[~in_group]

    if len(group_scores) < 2 or len(rest_scores) < 2:
        raise ValueError(
            f"Not enough scored rows to compare {group_value!r} "
            f"({len(group_scores)} in group, {len(rest_scores)} in rest)."
        )

    observed = group_scores.mean() - rest_scores.mean()
    rng = np.random.default_rng(seed)

    # Bootstrap: resample each side with replacement, keep the difference.
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = (
            rng.choice(group_scores, len(group_scores)).mean()
            - rng.choice(rest_scores, len(rest_scores)).mean()
        )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    # Permutation: shuffle the labels; how often is a gap this low by chance?
    shuffled = np.empty(n_resamples)
    n_group = len(group_scores)
    for i in range(n_resamples):
        mixed = rng.permutation(values)
        shuffled[i] = mixed[:n_group].mean() - mixed[n_group:].mean()
    p_lower = (np.sum(shuffled <= observed) + 1) / (n_resamples + 1)

    return {
        "group": group_value,
        "n_group": n_group,
        "n_rest": len(rest_scores),
        "mean_group": round(float(group_scores.mean()), 3),
        "mean_rest": round(float(rest_scores.mean()), 3),
        "difference": round(float(observed), 3),
        "ci95_low": round(float(ci_low), 3),
        "ci95_high": round(float(ci_high), 3),
        "p_value_lower": round(float(p_lower), 4),
    }


def describe_comparison(result):
    """One plain-language sentence for a product owner."""
    group = result["group"]

    if result["ci95_high"] < 0:
        verdict = (
            f"{group} prompts score lower, and the gap is unlikely to be chance "
            "(the 95% interval stays below zero)."
        )
    elif result["ci95_low"] > 0:
        verdict = (
            f"{group} prompts score higher, and the gap is unlikely to be chance "
            "(the 95% interval stays above zero)."
        )
    else:
        verdict = (
            f"We cannot say {group} prompts score differently: "
            "the 95% interval includes zero."
        )

    return (
        f"{verdict} Mean {result['mean_group']} vs {result['mean_rest']} "
        f"(difference {result['difference']}, 95% interval "
        f"{result['ci95_low']} to {result['ci95_high']}), based on "
        f"{result['n_group']} vs {result['n_rest']} scored calls."
    )


# ==================================================
# 5. Generate reports
# ==================================================

def main():
    df = load_data()

    print("API DATA ANALYSIS REPORT")
    print("=" * 60)
    print("Dataset:", INPUT_PATH.name)
    print("Total requests:", len(df))
    print("Models:", df["model"].nunique())
    print("Categories:", df["category"].nunique())

    overall_report = calculate_metrics(df)

    print("\nOVERALL METRICS")
    print("-" * 60)
    print(overall_report.to_string(index=False))

    model_report = calculate_metrics(df, "model")

    print("\nMETRICS BY MODEL")
    print("-" * 60)
    print(model_report.to_string(index=False))

    category_report = calculate_metrics(df, "category")

    print("\nMETRICS BY CATEGORY")
    print("-" * 60)
    print(category_report.to_string(index=False))

    print(
        "\nNote: group sizes and failure counts can be small. "
        "Compare counts, not only percentages."
    )

    silent_with_score = int(
        (df["is_silent_failure"] & df["judge_score"].notna()).sum()
    )
    print(
        f"Note: {silent_with_score} silent failures carry a judge_score. "
        "They are excluded from every judge-score number above."
    )

    comparison = compare_group_scores(df, "category", "adversarial")

    print("\nDO ADVERSARIAL PROMPTS SCORE LOWER?")
    print("-" * 60)
    print(pd.DataFrame([comparison]).to_string(index=False))
    print(describe_comparison(comparison))

    overall_report.to_csv(OVERALL_REPORT_PATH, index=False)
    model_report.to_csv(MODEL_REPORT_PATH, index=False)
    category_report.to_csv(CATEGORY_REPORT_PATH, index=False)
    pd.DataFrame([comparison]).to_csv(ADVERSARIAL_REPORT_PATH, index=False)

    print("\nREPORT FILES SAVED")
    print("-" * 60)
    print(OVERALL_REPORT_PATH.name)
    print(MODEL_REPORT_PATH.name)
    print(CATEGORY_REPORT_PATH.name)
    print(ADVERSARIAL_REPORT_PATH.name)

    print("\nReport generation completed successfully.")


if __name__ == "__main__":
    main()
