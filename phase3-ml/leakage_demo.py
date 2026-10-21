"""Step 2: show how much data leakage inflates the numbers."""
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ml_data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    load_data,
    require_columns,
)
from ml_model import build_model

LEAKY_NUMERIC = ["input_tokens", "output_tokens", "latency_ms", "judge_score", "response_length"]

# Why each Run 1 feature leaks ("would I know it BEFORE the call?")
LEAK_REASONS = {
    "output_tokens": "only exists after the model answered",
    "latency_ms": "measured during the call, so unknown before it",
    "judge_score": "produced by the judge AFTER reading the response",
    "response_length": "0 means blank response, which IS the silent-failure definition",
}


def score(df, numeric, categorical):
    """Return (test precision, test recall, cv precision, cv recall) for a feature set."""
    features = categorical + numeric
    X, y = df[features], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = build_model(numeric, categorical).fit(X_train, y_train)
    predictions = model.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_predictions = cross_val_predict(build_model(numeric, categorical), X, y, cv=cv)

    return (
        precision_score(y_test, predictions, zero_division=0),
        recall_score(y_test, predictions, zero_division=0),
        precision_score(y, cv_predictions, zero_division=0),
        recall_score(y, cv_predictions, zero_division=0),
    )


def run_leakage_demo(df=None) -> dict:
    df = load_data() if df is None else df
    require_columns(df, LEAKY_NUMERIC + CATEGORICAL_FEATURES)

    return {
        "rows": len(df),
        "failures": int(df[TARGET].sum()),
        "duplicate_prompts": int(df["prompt"].duplicated().sum()),
        "run1": score(df, LEAKY_NUMERIC, CATEGORICAL_FEATURES),
        "run2": score(df, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }


def main() -> None:
    r = run_leakage_demo()
    print(f"Rows: {r['rows']}  Failures: {r['failures']}")
    print(f"Duplicate prompts: {r['duplicate_prompts']}")
    if r["duplicate_prompts"] == 0:
        print("  -> every prompt is unique, so a grouped split is not needed.")
    else:
        print("  -> prompts repeat: use GroupShuffleSplit / GroupKFold by prompt.")

    header = f"{'':44} {'test P':>7} {'test R':>7} {'cv P':>7} {'cv R':>7}"
    print("\n" + header)
    for name, key in (
        ("Run 1: post-call features (LEAKAGE)", "run1"),
        ("Run 2: pre-call features (honest)", "run2"),
    ):
        p, rec, cvp, cvr = r[key]
        print(f"{name:44} {p:>7.3f} {rec:>7.3f} {cvp:>7.3f} {cvr:>7.3f}")

    print("\nWhy Run 1 leaks:")
    for feature, reason in LEAK_REASONS.items():
        print(f"  {feature}: {reason}")


if __name__ == "__main__":
    main()
